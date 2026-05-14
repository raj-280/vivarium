"""
ml_models/pspnet_model.py

PSPNet (Pyramid Scene Parsing Network) with ResNet50 backbone.
Used for pixel-level liquid segmentation in transparent vessels.

Architecture:
    ResNet50 (no pretrained weights) → PSP Module → Decoder → Binary mask

Output: per-pixel probability of liquid vs non-liquid.

Note: This model is always initialised with weights=None (random init).
      Your own trained checkpoint is loaded immediately after in
      pspnet_measurer.py via torch.load() + load_state_dict().
      ImageNet pretraining is never used at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module.
    Pools features at 4 scales, applies conv, upsamples, concatenates.
    """

    def __init__(self, in_channels: int = 2048, pool_sizes: list = [1, 2, 3, 6]) -> None:
        super().__init__()
        out_channels = in_channels // len(pool_sizes)

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        pyramids = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=True)
            pyramids.append(upsampled)
        return torch.cat(pyramids, dim=1)


class PSPNet(nn.Module):
    """
    PSPNet with ResNet50 backbone for binary liquid segmentation.

    Input:  BGR image tensor (B, 3, H, W) — normalised
    Output: (B, 2, H, W) logits — channel 0 = background, channel 1 = liquid
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        # --- ResNet50 encoder ---
        # weights=None — random init only. Your own checkpoint is loaded
        # immediately after construction in pspnet_measurer.py.
        resnet = models.resnet50(weights=None)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4,  channels 256
        self.layer2 = resnet.layer2   # stride 8,  channels 512
        self.layer3 = resnet.layer3   # stride 16, channels 1024
        self.layer4 = resnet.layer4   # stride 32, channels 2048

        # --- PSP module ---
        # in_channels=2048, pool_sizes=[1,2,3,6] → out = 2048 + 4*(2048//4) = 4096
        self.psp = PSPModule(in_channels=2048, pool_sizes=[1, 2, 3, 6])

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = (x.shape[2], x.shape[3])

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.psp(x)
        x = self.decoder(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x
