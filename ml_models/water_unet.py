"""
pipeline/neural/water_unet.py

Tier 2 — Neural refinement for water level.

Architecture (per spec doc):
  - Input:        warped tube crop resized to 128×256 (W×H)
  - Architecture: 3 encoder blocks (32→64→128 channels), 3 decoder blocks
  - Output:       Binary segmentation mask (sigmoid) of water fill region
  - Loss:         BCE + Dice
  - Optimiser:    AdamW + cosine LR schedule
  - Annotation:   300 tube masks (fill line is obvious → cheaply labelable)

Inference usage:
    model = WaterUNet()
    model.load_state_dict(torch.load("weights/water_unet.pt"))
    model.eval()

    # roi is a BGR numpy array (tube crop)
    mask, water_pct, confidence = WaterUNetInferencer(model).predict(roi)

Training usage:
    trainer = WaterUNetTrainer(
        train_dir="data/water/train",
        val_dir="data/water/val",
        output_dir="models/weights",
    )
    trainer.train(epochs=50)

Dataset folder structure expected by trainer:
    data/water/train/
        images/   ← tube crop JPEGs (any size, resized to 128×256 at load time)
        masks/    ← binary PNG masks (255=water, 0=air), same filenames as images
    data/water/val/
        images/
        masks/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from loguru import logger

# Input dimensions (W×H) per architecture doc
WATER_UNET_W = 128
WATER_UNET_H = 256


# ===========================================================================
# Model Architecture
# ===========================================================================

class _ConvBlock(nn.Module):
    """Double Conv → BN → ReLU block used in encoder and decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.block(x)


class WaterUNet(nn.Module):
    """
    3-block U-Net for binary segmentation of the water fill region.

    Encoder: 3→32→64→128 channels, MaxPool between blocks
    Bottleneck: 128→256
    Decoder: 256→128→64→32, bilinear up + skip connections
    Output: 1-channel sigmoid mask

    Input:  (B, 3, H=256, W=128) — RGB, normalised to [0,1]
    Output: (B, 1, H=256, W=128) — sigmoid probability mask
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.enc1 = _ConvBlock(3, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)

        # Bottleneck
        self.bottleneck = _ConvBlock(128, 256)

        # Decoder — receives upsampled + skip concat
        self.dec3 = _ConvBlock(256 + 128, 128)
        self.dec2 = _ConvBlock(128 + 64, 64)
        self.dec1 = _ConvBlock(64 + 32, 32)

        # Output head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Encoder
        e1 = self.enc1(x)                     # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))         # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))         # (B, 128, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))    # (B, 256, H/8, W/8)

        # Decoder with skip connections
        d3 = self._up(b, e3)
        d3 = self.dec3(d3)                    # (B, 128, H/4, W/4)

        d2 = self._up(d3, e2)
        d2 = self.dec2(d2)                    # (B, 64, H/2, W/2)

        d1 = self._up(d2, e1)
        d1 = self.dec1(d1)                    # (B, 32, H, W)

        return torch.sigmoid(self.head(d1))   # (B, 1, H, W) — sigmoid output

    @staticmethod
    def _up(
        x: "torch.Tensor", skip: "torch.Tensor"
    ) -> "torch.Tensor":
        """Bilinear upsample x to match skip, then concatenate."""
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


# ===========================================================================
# Loss Functions
# ===========================================================================

class BCEDiceLoss(nn.Module):
    """
    BCE + Dice loss combination per architecture doc.

    Dice loss handles class imbalance (water region << air region in near-empty bottles).
    BCE provides pixel-wise gradient signal.

    Args:
        bce_weight:  Weight for BCE term. Default 0.5.
        dice_weight: Weight for Dice term. Default 0.5.
        smooth:      Smoothing constant to prevent division by zero.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self._bce_w = bce_weight
        self._dice_w = dice_weight
        self._smooth = smooth
        self._bce = nn.BCELoss()

    def forward(
        self, pred: "torch.Tensor", target: "torch.Tensor"
    ) -> "torch.Tensor":
        # BCE
        bce_loss = self._bce(pred, target)

        # Dice
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self._smooth) / (
            pred_flat.sum() + target_flat.sum() + self._smooth
        )

        return self._bce_w * bce_loss + self._dice_w * dice_loss


# ===========================================================================
# Dataset
# ===========================================================================

class WaterTubeDataset(Dataset):
    """
    PyTorch Dataset for water tube crops + binary fill masks.

    Folder structure:
        root/images/  ← JPEGs (tube crops, any resolution)
        root/masks/   ← PNG masks (255=water, 0=air), same stem as images

    Images are resized to WATER_UNET_W × WATER_UNET_H (128×256) at load time.
    """

    def __init__(self, root: str | Path, augment: bool = False) -> None:
        self._img_dir = Path(root) / "images"
        self._mask_dir = Path(root) / "masks"

        stems = sorted([p.stem for p in self._img_dir.glob("*.jpg")])
        stems += sorted([p.stem for p in self._img_dir.glob("*.png")])
        self._stems = sorted(set(stems))

        if not self._stems:
            raise FileNotFoundError(
                f"No images found in {self._img_dir}. "
                "Expected JPG or PNG files."
            )

        logger.info(f"WaterTubeDataset: {len(self._stems)} samples in {root}")

        # Image transform: resize → tensor → normalise
        self._img_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((WATER_UNET_H, WATER_UNET_W)),
            transforms.ToTensor(),                    # [0,1] float32
            transforms.Normalize([0.5, 0.5, 0.5],    # rough normalisation
                                  [0.5, 0.5, 0.5]),
        ])

        # Augmentation (training only)
        self._augment = augment
        self._aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.Resize((WATER_UNET_H, WATER_UNET_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Mask transform: resize → tensor (no normalise)
        self._mask_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (WATER_UNET_H, WATER_UNET_W),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),  # [0,1] float32
        ])

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int):
        import cv2

        stem = self._stems[idx]

        # Load image (try jpg then png)
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = self._img_dir / f"{stem}{ext}"
            if img_path.exists():
                break
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Could not read image: {img_path}")
        img_rgb = img_bgr[:, :, ::-1].copy()

        # Load mask
        mask_path = self._mask_dir / f"{stem}.png"
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise IOError(f"Could not read mask: {mask_path}")

        # Binarise mask (255 → 1.0)
        mask_bin = (mask_gray > 127).astype(np.uint8) * 255

        # Apply transforms
        if self._augment:
            img_tensor = self._aug_tf(img_rgb)
        else:
            img_tensor = self._img_tf(img_rgb)

        mask_tensor = self._mask_tf(mask_bin)  # (1, H, W)

        return img_tensor, mask_tensor


# ===========================================================================
# Trainer
# ===========================================================================

class WaterUNetTrainer:
    """
    Full training loop for WaterUNet.

    Implements:
      - AdamW optimiser (per architecture doc)
      - Cosine LR schedule (per architecture doc)
      - BCE + Dice loss (per architecture doc)
      - Best-model checkpointing by validation Dice score
      - Saves final weights to output_dir/water_unet.pt

    Args:
        train_dir:   Path to training data (images/ + masks/ subdirs).
        val_dir:     Path to validation data.
        output_dir:  Where to save the trained model weights.
        device:      "cuda" | "cpu" | "mps". Auto-detected if None.
        batch_size:  Training batch size. Default 8.
        lr:          Initial learning rate for AdamW. Default 1e-3.
        epochs:      Number of training epochs. Default 50.
    """

    def __init__(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        output_dir: str | Path = "models/weights",
        device: Optional[str] = None,
        batch_size: int = 8,
        lr: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for training. Run: pip install torch torchvision")

        self._train_dir = Path(train_dir)
        self._val_dir = Path(val_dir)
        self._output_dir = Path(output_dir)
        self._batch_size = batch_size
        self._lr = lr
        self._epochs = epochs

        # Auto device detection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)
        logger.info(f"WaterUNetTrainer using device: {self._device}")

    def train(self) -> WaterUNet:
        """Run the full training loop. Returns the best-performing model."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        train_ds = WaterTubeDataset(self._train_dir, augment=True)
        val_ds = WaterTubeDataset(self._val_dir, augment=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self._device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=
            0),

        # Model, loss, optimiser, scheduler
        model = WaterUNet().to(self._device)
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

        # AdamW per architecture doc
        optimiser = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=1e-4)

        # Cosine LR schedule per architecture doc
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self._epochs, eta_min=1e-6
        )

        best_dice = -1.0
        best_path = self._output_dir / "water_unet.pt"

        for epoch in range(1, self._epochs + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs = imgs.to(self._device)
                masks = masks.to(self._device)

                optimiser.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimiser.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)
            scheduler.step()

            # --- Validate ---
            val_dice = self._validate(model, val_loader)

            current_lr = optimiser.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:03d}/{self._epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_dice={val_dice:.4f} | "
                f"lr={current_lr:.2e}"
            )

            # Checkpoint best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), best_path)
                logger.info(f"  ✓ New best model saved (dice={best_dice:.4f}) → {best_path}")

        logger.info(f"Training complete. Best val Dice: {best_dice:.4f}")
        logger.info(f"Weights saved to: {best_path}")

        # Load best weights back
        model.load_state_dict(torch.load(best_path, map_location=self._device))
        return model

    def _validate(self, model: WaterUNet, loader: DataLoader) -> float:
        """Compute mean Dice score over the validation set."""
        model.eval()
        dice_scores = []
        smooth = 1.0

        with torch.no_grad():
            for imgs, masks in loader:
                imgs = imgs.to(self._device)
                masks = masks.to(self._device)
                preds = model(imgs)

                # Threshold at 0.5 for binary Dice
                preds_bin = (preds > 0.5).float()
                intersection = (preds_bin * masks).sum(dim=(1, 2, 3))
                dice = (2.0 * intersection + smooth) / (
                    preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + smooth
                )
                dice_scores.extend(dice.cpu().tolist())

        return float(np.mean(dice_scores)) if dice_scores else 0.0


# ===========================================================================
# Inferencer — used by opencv_water_measurer.py at runtime
# ===========================================================================

class WaterUNetInferencer:
    """
    Wraps a trained WaterUNet for inference.

    Takes a BGR numpy array (tube crop), returns:
      - mask:       Binary numpy array (H×W) of the fill region
      - water_pct:  Fill percentage derived from mask
      - confidence: Mean sigmoid probability in the fill region (proxy for certainty)

    Usage:
        inferencer = WaterUNetInferencer.from_weights("models/weights/water_unet.pt")
        mask, water_pct, confidence = inferencer.predict(roi_bgr)
    """

    def __init__(self, model: "WaterUNet", device: str = "cpu") -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required. Run: pip install torch torchvision")
        self._model = model
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((WATER_UNET_H, WATER_UNET_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_weights(
        cls, weights_path: str | Path, device: str = "cpu"
    ) -> "WaterUNetInferencer":
        """Load from saved state dict."""
        model = WaterUNet()
        model.load_state_dict(
            torch.load(str(weights_path), map_location=device)
        )
        return cls(model, device=device)

    def predict(
        self,
        roi_bgr: np.ndarray,
        tube_top_y: int = 0,
        min_fill_rows: int = 3,
        center_band_frac: float = 0.6,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Run inference on a BGR tube crop.

        Args:
            roi_bgr:           BGR numpy array — warped tube crop.
            tube_top_y:        Pixel row (in roi coords) where the usable fill
                               zone begins — i.e. the top of the tube interior,
                               below any cap, fitting, or label. Rows above
                               this line are ignored when computing water_pct.
                               Default 0 (use full ROI height). Set this in
                               config as water.unet_water.tube_top_y.
            min_fill_rows:     Minimum consecutive filled rows required before a
                               fill region is accepted. Filters out single stray
                               noise pixels that would otherwise snap y_fill_top
                               to the top of the mask and report ~100% falsely.
                               Default 3.
            center_band_frac:  Fraction of the mask WIDTH (centred) used when
                               deciding which rows count as "filled". A row is
                               filled only if >=1 pixel in this central band is
                               water, which ignores tube-wall pixels on the far
                               left/right edges the model sometimes mis-segments.
                               Default 0.6 (middle 60%).

        Returns:
            mask:       (H_orig, W_orig) binary uint8 array — 255=water, 0=air
            water_pct:  Fill percentage 0-100, measured relative to the usable
                        tube height (orig_h - tube_top_y), NOT the whole ROI.
            confidence: 0-1 proxy confidence (mean sigmoid prob in fill region)
        """
        import cv2

        orig_h, orig_w = roi_bgr.shape[:2]

        # Clamp tube_top_y to a valid range
        tube_top_y = int(np.clip(tube_top_y, 0, orig_h - 1))
        usable_h = orig_h - tube_top_y  # the span we actually measure against

        # Convert BGR → RGB, preprocess
        roi_rgb = roi_bgr[:, :, ::-1].copy()
        tensor = self._preprocess(roi_rgb).unsqueeze(0).to(self._device)  # (1,3,H,W)

        with torch.no_grad():
            prob_map = self._model(tensor)  # (1,1,H,W) sigmoid

        prob_np = prob_map.squeeze().cpu().numpy()  # (H,W) float [0,1]

        # Threshold → binary mask, resize back to original roi dimensions
        mask_small = (prob_np > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(
            mask_small, (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # FIX 1: ignore rows above the usable fill zone (cap, fitting, label area)
        usable_mask = mask[tube_top_y:, :]
        usable_prob = prob_np[tube_top_y:, :]

        # FIX 2: use only the centre band to detect filled rows
        # Tube walls on left/right edges are sometimes mis-segmented as water —
        # they run full height, causing y_fill_top=0 and a false 100% reading.
        band_start = int(orig_w * (1 - center_band_frac) / 2)
        band_end   = max(int(orig_w * (1 + center_band_frac) / 2), band_start + 1)
        center_band = usable_mask[:, band_start:band_end]
        row_fill = np.any(center_band > 127, axis=1)  # (usable_h,) bool

        # FIX 3: require min_fill_rows consecutive filled rows
        # A single stray pixel causes filled_rows.min()=0 → false 100%.
        # Find the topmost run of at least min_fill_rows consecutive rows instead.
        filled_indices = np.where(row_fill)[0]

        if len(filled_indices) == 0:
            water_pct  = 0.0
            confidence = float(1.0 - np.mean(usable_prob))
        else:
            y_fill_top_local = None
            run_start = filled_indices[0]
            run_len   = 1
            for i in range(1, len(filled_indices)):
                if filled_indices[i] == filled_indices[i - 1] + 1:
                    run_len += 1
                    if run_len >= min_fill_rows and y_fill_top_local is None:
                        y_fill_top_local = run_start
                        break
                else:
                    run_start = filled_indices[i]
                    run_len   = 1

            # No qualifying run → genuine near-empty, use the lowest isolated pixel
            if y_fill_top_local is None:
                y_fill_top_local = int(filled_indices[-1])

            # FIX 1 applied: divide by usable_h, not orig_h
            water_pct = float((usable_h - y_fill_top_local) / usable_h * 100.0)
            water_pct = float(np.clip(water_pct, 0.0, 100.0))

            fill_region = usable_prob[y_fill_top_local:, :]
            confidence  = float(np.mean(fill_region)) if fill_region.size > 0 else 0.5

        return mask, round(water_pct, 1), float(np.clip(confidence, 0.0, 1.0))