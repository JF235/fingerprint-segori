from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SameMaxPool2d


class SnfenNet(nn.Module):
    """PyTorch port of SNFEN model used in pyfing.enhancement.Snfen."""

    def __init__(self):
        super().__init__()
        self._keras_layer_map: dict[str, nn.Module] = {}

        in_ch = 5
        for i in range(5):
            out_ch = 16 * (2**i)
            conv_name = f"enc_{i}_conv"
            bn_name = f"enc_{i}_bn"
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=True)
            bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.99)
            setattr(self, conv_name, conv)
            setattr(self, bn_name, bn)
            self._keras_layer_map[conv_name] = conv
            self._keras_layer_map[bn_name] = bn
            in_ch = out_ch

        dec_in = [256, 512, 256, 128, 64]
        dec_out = [256, 128, 64, 32, 16]
        for i, (in_c, out_c) in enumerate(zip(dec_in, dec_out)):
            conv_name = f"dec_{i}_conv"
            bn_name = f"dec_{i}_bn"
            conv = nn.Conv2d(in_c, out_c, kernel_size=5, padding=2, bias=True)
            bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.99)
            setattr(self, conv_name, conv)
            setattr(self, bn_name, bn)
            self._keras_layer_map[conv_name] = conv
            self._keras_layer_map[bn_name] = bn

        self.head_conv = nn.Conv2d(32, 16, kernel_size=5, padding=2, bias=True)
        self.head_bn = nn.BatchNorm2d(16, eps=1e-3, momentum=0.99)
        self.head_conv_final = nn.Conv2d(16, 1, kernel_size=5, padding=2, bias=True)
        self._keras_layer_map["head_conv"] = self.head_conv
        self._keras_layer_map["head_bn"] = self.head_bn
        self._keras_layer_map["head_conv_final"] = self.head_conv_final

        self.pool = SameMaxPool2d(kernel_size=2, stride=2)

    def keras_layer_map(self) -> dict[str, nn.Module]:
        return self._keras_layer_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,4,H,W] = fingerprint, mask, orientation_q, ridge_period_q
        x = x.float()
        fingerprints = x[:, 0:1]
        masks = x[:, 1:2]
        rad2 = x[:, 2:3] / 255.0 * (2.0 * np.pi)
        sin2 = torch.sin(rad2)
        cos2 = torch.cos(rad2)
        rp = x[:, 3:4] / 10.0
        x = torch.cat([fingerprints, masks, sin2, cos2, rp], dim=1)

        skips: list[torch.Tensor] = []
        for i in range(5):
            conv = getattr(self, f"enc_{i}_conv")
            bn = getattr(self, f"enc_{i}_bn")
            x = bn(F.relu(conv(x)))
            skips.append(x)
            x = self.pool(x)

        for i in range(5):
            conv = getattr(self, f"dec_{i}_conv")
            bn = getattr(self, f"dec_{i}_bn")
            x = bn(F.relu(conv(x)))
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        x = self.head_bn(F.relu(self.head_conv(x)))
        x = torch.sigmoid(self.head_conv_final(x))
        return x[:, 0]
