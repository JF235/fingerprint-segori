from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SameMaxPool2d


class SnffeNet(nn.Module):
    """PyTorch port of SNFFE model used in pyfing.frequencies.Snffe."""

    def __init__(self):
        super().__init__()
        self._keras_layer_map: dict[str, nn.Module] = {}

        in_ch = 4
        for i in range(5):
            out_ch = 16 * (2**i)
            conv_name = f"enc_{i}_conv_5"
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
            conv_name = f"dec_{i}_conv_5"
            bn_name = f"dec_{i}_bn"
            conv = nn.Conv2d(in_c, out_c, kernel_size=5, padding=2, bias=True)
            bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.99)
            setattr(self, conv_name, conv)
            setattr(self, bn_name, bn)
            self._keras_layer_map[conv_name] = conv
            self._keras_layer_map[bn_name] = bn

        self.head_0_conv_5 = nn.Conv2d(32, 16, kernel_size=5, padding=2, bias=True)
        self.head_0_bn = nn.BatchNorm2d(16, eps=1e-3, momentum=0.99)
        self.head_linear_conv_3 = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)
        self._keras_layer_map["head_0_conv_5"] = self.head_0_conv_5
        self._keras_layer_map["head_0_bn"] = self.head_0_bn
        self._keras_layer_map["head_linear_conv_3"] = self.head_linear_conv_3

        self.pool = SameMaxPool2d(kernel_size=2, stride=2)

    def keras_layer_map(self) -> dict[str, nn.Module]:
        return self._keras_layer_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] = fingerprint, mask, orientation (quantized 0..255)
        x = x.float()
        rad = x[:, 2:3] / 255.0 * (2.0 * np.pi)
        sin2 = torch.sin(rad)
        cos2 = torch.cos(rad)
        x = torch.cat([x[:, 0:2], sin2, cos2], dim=1)

        skips: list[torch.Tensor] = []
        for i in range(5):
            conv = getattr(self, f"enc_{i}_conv_5")
            bn = getattr(self, f"enc_{i}_bn")
            x = bn(F.relu(conv(x)))
            skips.append(x)
            x = self.pool(x)

        for i in range(5):
            conv = getattr(self, f"dec_{i}_conv_5")
            bn = getattr(self, f"dec_{i}_bn")
            x = bn(F.relu(conv(x)))
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        x = self.head_0_bn(F.relu(self.head_0_conv_5(x)))
        x = self.head_linear_conv_3(x)
        return x[:, 0]
