from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SameMaxPool2d


class SufsNet(nn.Module):
    """PyTorch port of SUFS model used in pyfing.segmentation.Sufs."""

    def __init__(self):
        super().__init__()
        self._keras_layer_map: dict[str, nn.Module] = {}

        filters = [16, 32, 64, 128, 256, 512]

        in_ch = 1
        for i, out_ch in enumerate(filters):
            conv_name = f"conv2d{'' if i == 0 else f'_{i}'}"
            bn_name = f"batch_normalization{'' if i == 0 else f'_{i}'}"
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
            bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.99)
            setattr(self, conv_name, conv)
            setattr(self, bn_name, bn)
            self._keras_layer_map[conv_name] = conv
            self._keras_layer_map[bn_name] = bn
            in_ch = out_ch

        dec_in_channels = [512, 1024, 512, 256, 128, 64]
        for i, (in_decoder, out_ch) in enumerate(zip(dec_in_channels, reversed(filters))):
            conv_idx = len(filters) + i
            conv_name = f"conv2d_{conv_idx}"
            bn_name = f"batch_normalization_{conv_idx}"
            conv = nn.Conv2d(in_decoder, out_ch, kernel_size=3, padding=1, bias=True)
            bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.99)
            setattr(self, conv_name, conv)
            setattr(self, bn_name, bn)
            self._keras_layer_map[conv_name] = conv
            self._keras_layer_map[bn_name] = bn

        self.conv2d_12 = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True)
        self._keras_layer_map["conv2d_12"] = self.conv2d_12

        self.pool = SameMaxPool2d(kernel_size=2, stride=2)

    def keras_layer_map(self) -> dict[str, nn.Module]:
        return self._keras_layer_map

    def _enc_block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        conv = getattr(self, f"conv2d{'' if i == 0 else f'_{i}'}")
        bn = getattr(self, f"batch_normalization{'' if i == 0 else f'_{i}'}")
        x = conv(x)
        x = F.relu(x)
        x = bn(x)
        return x

    def _dec_block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        conv_idx = 6 + i
        conv = getattr(self, f"conv2d_{conv_idx}")
        bn = getattr(self, f"batch_normalization_{conv_idx}")
        x = conv(x)
        x = F.relu(x)
        x = bn(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected in range [0, 255], shape [B, 1, H, W].
        levels: list[torch.Tensor] = []
        for i in range(6):
            x = self._enc_block(x, i)
            levels.append(x)
            x = self.pool(x)

        for i, skip in enumerate(reversed(levels)):
            x = self._dec_block(x, i)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        x = self.conv2d_12(x)
        return torch.sigmoid(x)
