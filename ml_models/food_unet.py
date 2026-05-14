"""
pipeline/neural/food_unet.py

Tier 2 — Neural refinement for food level.

Architecture (per spec doc):
  - Input:        hopper crop resized to 256×256
  - Architecture: same 3-block U-Net structure as water_unet
  - Output:       surface_y SCALAR (regression, NOT segmentation)
                  This is the key difference from water_unet — food surface
                  is a single y-coordinate, not a pixel mask.
  - Loss:         MSE + smoothness regulariser (L1 on predicted y deviation)
  - Optimiser:    AdamW + cosine LR schedule
  - Annotation:   300 hopper images with surface_y labeled (pixel row)

Why regression instead of segmentation?
  The food surface is an irregular pile — not a clean horizontal fill line
  like water. A single scalar y-coordinate describing "where the surface is"
  is more useful and cheaper to annotate than pixel masks.

Annotation format (CSV, one row per image):
    filename,surface_y,image_height
    hopper_001.jpg,142,256
    hopper_002.jpg,89,256
    ...
    surface_y is the pixel row of the food surface (0 = top, H = bottom).

Inference usage:
    inferencer = FoodUNetInferencer.from_weights("models/weights/food_unet.pt")
    surface_y, food_pct, confidence = inferencer.predict(roi_bgr)

Training usage:
    trainer = FoodUNetTrainer(
        train_csv="data/food/train_labels.csv",
        train_img_dir="data/food/train/images",
        val_csv="data/food/val_labels.csv",
        val_img_dir="data/food/val/images",
        output_dir="models/weights",
    )
    trainer.train(epochs=50)
"""

from __future__ import annotations

import csv
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

# Input dimensions per architecture doc
FOOD_UNET_SIZE = 256   # square: 256×256


# ===========================================================================
# Shared Conv Block (same as water_unet)
# ===========================================================================

class _ConvBlock(nn.Module):
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

    def forward(self, x):
        return self.block(x)


# ===========================================================================
# Model Architecture
# ===========================================================================

class FoodUNet(nn.Module):
    """
    3-block U-Net for food surface regression.

    Same encoder/decoder structure as WaterUNet, but the output head
    produces a single scalar (normalised surface_y ∈ [0,1]) instead of
    a segmentation mask.

    Encoder: 3→32→64→128 channels, MaxPool between blocks
    Bottleneck: 128→256
    Decoder: 256→128→64→32, bilinear up + skip connections
    Regression head: global average pool → FC(32→1) → sigmoid

    The sigmoid output maps to [0,1]:
      - 0.0 = surface at top of hopper (completely full)
      - 1.0 = surface at bottom (completely empty)

    Input:  (B, 3, 256, 256)
    Output: (B, 1) — normalised surface_y prediction
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder (identical to WaterUNet)
        self.enc1 = _ConvBlock(3, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)

        # Bottleneck
        self.bottleneck = _ConvBlock(128, 256)

        # Decoder
        self.dec3 = _ConvBlock(256 + 128, 128)
        self.dec2 = _ConvBlock(128 + 64, 64)
        self.dec1 = _ConvBlock(64 + 32, 32)

        # Regression head:
        # Global average pool over spatial dims → flatten → FC → sigmoid
        # This aggregates spatial context into a single surface estimate
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # (B, 32, 1, 1)
            nn.Flatten(),                # (B, 32)
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),                # output ∈ [0, 1]
        )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], dim=1))

        # Regression: scalar surface_y ∈ [0, 1]
        return self.regressor(d1)   # (B, 1)


# ===========================================================================
# Loss Function
# ===========================================================================

class SurfaceRegressionLoss(nn.Module):
    """
    MSE loss for surface_y regression, with an optional smoothness term.

    MSE: primary regression signal
    L1 smoothness: penalises predictions far from 0.5 when confidence is low
                   (prevents the model from predicting extreme values on
                   ambiguous frames like completely full or occluded hoppers)

    Args:
        mse_weight:       Weight for MSE term. Default 1.0.
        smooth_weight:    Weight for L1 regularisation. Default 0.1.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        smooth_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self._mse_w = mse_weight
        self._smooth_w = smooth_weight

    def forward(
        self, pred: "torch.Tensor", target: "torch.Tensor"
    ) -> "torch.Tensor":
        # pred, target: (B, 1) normalised surface_y

        # Primary: MSE
        mse = F.mse_loss(pred, target)

        # Smoothness: L1 penalty on deviation from centre (0.5)
        # Prevents extreme predictions on near-empty or near-full hoppers
        smooth = torch.mean(torch.abs(pred - 0.5))

        return self._mse_w * mse + self._smooth_w * smooth


# ===========================================================================
# Dataset
# ===========================================================================

class FoodHopperDataset(Dataset):
    """
    PyTorch Dataset for food hopper crops + surface_y labels.

    Annotation CSV format (one row per image):
        filename,surface_y,image_height
        hopper_001.jpg,142,256

    surface_y is stored as a raw pixel row — the dataset normalises it to [0,1]
    by dividing by image_height.

    Images are resized to FOOD_UNET_SIZE × FOOD_UNET_SIZE at load time.

    Args:
        csv_path:  Path to the annotation CSV.
        img_dir:   Directory containing the image files listed in the CSV.
        augment:   Apply random augmentation (use True for training only).
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self._img_dir = Path(img_dir)
        self._augment = augment
        self._samples = []  # list of (filename, surface_y_norm)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                surface_y = float(row["surface_y"])
                img_height = float(row.get("image_height", FOOD_UNET_SIZE))
                # Normalise: 0 = top (full), 1 = bottom (empty)
                surface_y_norm = float(np.clip(surface_y / img_height, 0.0, 1.0))
                self._samples.append((filename, surface_y_norm))

        logger.info(f"FoodHopperDataset: {len(self._samples)} samples from {csv_path}")

        self._img_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((FOOD_UNET_SIZE, FOOD_UNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self._aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.Resize((FOOD_UNET_SIZE, FOOD_UNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        import cv2

        filename, surface_y_norm = self._samples[idx]

        img_path = self._img_dir / filename
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Could not read image: {img_path}")
        img_rgb = img_bgr[:, :, ::-1].copy()

        if self._augment:
            img_tensor = self._aug_tf(img_rgb)
        else:
            img_tensor = self._img_tf(img_rgb)

        label = torch.tensor([surface_y_norm], dtype=torch.float32)
        return img_tensor, label


# ===========================================================================
# Trainer
# ===========================================================================

class FoodUNetTrainer:
    """
    Full training loop for FoodUNet.

    Implements:
      - AdamW optimiser (per architecture doc)
      - Cosine LR schedule (per architecture doc)
      - MSE + smoothness loss
      - Best-model checkpointing by validation MAE
      - Saves to output_dir/food_unet.pt

    Args:
        train_csv:    Path to training annotation CSV.
        train_img_dir: Directory with training images.
        val_csv:      Path to validation annotation CSV.
        val_img_dir:  Directory with validation images.
        output_dir:   Where to save weights.
        device:       "cuda" | "cpu" | "mps". Auto-detected if None.
        batch_size:   Default 16 (regression is cheaper than segmentation).
        lr:           Initial AdamW LR. Default 1e-3.
        epochs:       Training epochs. Default 50.
    """

    def __init__(
        self,
        train_csv: str | Path,
        train_img_dir: str | Path,
        val_csv: str | Path,
        val_img_dir: str | Path,
        output_dir: str | Path = "models/weights",
        device: Optional[str] = None,
        batch_size: int = 16,
        lr: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required. Run: pip install torch torchvision")

        self._train_csv = Path(train_csv)
        self._train_img_dir = Path(train_img_dir)
        self._val_csv = Path(val_csv)
        self._val_img_dir = Path(val_img_dir)
        self._output_dir = Path(output_dir)
        self._batch_size = batch_size
        self._lr = lr
        self._epochs = epochs

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)
        logger.info(f"FoodUNetTrainer using device: {self._device}")

    def train(self) -> FoodUNet:
        """Run the full training loop. Returns the best-performing model."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        train_ds = FoodHopperDataset(self._train_csv, self._train_img_dir, augment=True)
        val_ds = FoodHopperDataset(self._val_csv, self._val_img_dir, augment=False)

        import os
        train_loader = DataLoader(
            train_ds, batch_size=self._batch_size, shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(self._device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self._batch_size, shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
        )

        model = FoodUNet().to(self._device)
        criterion = SurfaceRegressionLoss(mse_weight=1.0, smooth_weight=0.1)

        # AdamW per architecture doc
        optimiser = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=1e-4)

        # Cosine LR schedule per architecture doc
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self._epochs, eta_min=1e-6
        )

        best_mae = float("inf")
        best_path = self._output_dir / "food_unet.pt"

        for epoch in range(1, self._epochs + 1):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for imgs, labels in train_loader:
                imgs = imgs.to(self._device)
                labels = labels.to(self._device)

                optimiser.zero_grad()
                preds = model(imgs)          # (B, 1)
                loss = criterion(preds, labels)
                loss.backward()
                optimiser.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)
            scheduler.step()

            # --- Validate ---
            val_mae = self._validate(model, val_loader)

            # MAE is in normalised [0,1] units — convert to percentage points
            val_mae_pct = val_mae * 100.0
            current_lr = optimiser.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d}/{self._epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_MAE={val_mae_pct:.2f}% | "
                f"lr={current_lr:.2e}"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), best_path)
                logger.info(
                    f"  ✓ New best model (MAE={val_mae_pct:.2f}%) → {best_path}"
                )

        logger.info(
            f"Training complete. Best val MAE: {best_mae * 100.0:.2f}%"
        )
        model.load_state_dict(torch.load(best_path, map_location=self._device))
        return model

    def _validate(self, model: FoodUNet, loader: DataLoader) -> float:
        """Compute mean absolute error over the validation set (normalised units)."""
        model.eval()
        errors = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self._device)
                labels = labels.to(self._device)
                preds = model(imgs)
                mae = torch.abs(preds - labels).mean().item()
                errors.append(mae)
        return float(np.mean(errors)) if errors else 0.0


# ===========================================================================
# Inferencer
# ===========================================================================

class FoodUNetInferencer:
    """
    Wraps a trained FoodUNet for inference.

    Returns:
      - surface_y:  Pixel row of food surface in the original ROI coordinate space
      - food_pct:   Fill percentage 0–100
      - confidence: Proxy confidence derived from prediction stability

    Usage:
        inf = FoodUNetInferencer.from_weights("models/weights/food_unet.pt")
        surface_y, food_pct, confidence = inf.predict(roi_bgr)
    """

    def __init__(self, model: "FoodUNet", device: str = "cpu") -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required.")
        self._model = model
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((FOOD_UNET_SIZE, FOOD_UNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_weights(
        cls, weights_path: str | Path, device: str = "cpu"
    ) -> "FoodUNetInferencer":
        model = FoodUNet()
        model.load_state_dict(
            torch.load(str(weights_path), map_location=device)
        )
        return cls(model, device=device)

    def predict(
        self, roi_bgr: np.ndarray, n_passes: int = 5
    ) -> Tuple[int, float, float]:
        """
        Run inference with Monte Carlo dropout for confidence estimation.

        Args:
            roi_bgr:  BGR numpy array — food hopper crop.
            n_passes: Number of forward passes with dropout enabled.
                      More passes = better confidence estimate, slower inference.
                      Set n_passes=1 to disable MC dropout (fastest).

        Returns:
            surface_y:  Pixel row of food surface in original roi coordinate space.
            food_pct:   Fill percentage 0–100.
            confidence: 0–1 confidence (inverse of prediction std across passes).
        """
        orig_h = roi_bgr.shape[0]

        roi_rgb = roi_bgr[:, :, ::-1].copy()
        tensor = self._preprocess(roi_rgb).unsqueeze(0).to(self._device)

        if n_passes == 1:
            # Single pass — no MC dropout
            with torch.no_grad():
                pred_norm = self._model(tensor).squeeze().item()
            std_norm = 0.05  # default uncertainty without MC
        else:
            # Monte Carlo dropout: enable dropout at inference for uncertainty
            self._model.train()   # enables dropout
            preds = []
            with torch.no_grad():
                for _ in range(n_passes):
                    p = self._model(tensor).squeeze().item()
                    preds.append(p)
            self._model.eval()

            pred_norm = float(np.mean(preds))
            std_norm = float(np.std(preds))

        # Denormalise: surface_y in original roi pixel space
        surface_y = int(np.clip(pred_norm * orig_h, 0, orig_h - 1))

        # food_pct = (bot_y - surface_y) / hopper_height × 100 (architecture doc formula)
        food_pct = float(np.clip((orig_h - surface_y) / orig_h * 100.0, 0.0, 100.0))

        # Confidence: inverse of normalised std (lower variance → higher confidence)
        # std of 0 = perfect confidence (1.0), std of 0.1+ = low confidence
        confidence = float(np.clip(1.0 - std_norm * 10.0, 0.1, 1.0))

        return surface_y, round(food_pct, 1), confidence