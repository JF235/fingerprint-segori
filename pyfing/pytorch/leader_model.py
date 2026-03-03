from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ChannelLayerNorm, SameAvgPool2d, SameMaxPool2d, SeparableConv2d


class LeaderNet(nn.Module):
    """PyTorch port of LEADER model used in pyfing.minutiae.Leader."""

    def __init__(self):
        super().__init__()
        self._keras_layer_map: dict[str, nn.Module] = {}

        self.maxpool2 = SameMaxPool2d(kernel_size=2, stride=2)
        self.avgpool2 = SameAvgPool2d(kernel_size=2, stride=2)
        self.maxpool7_stride1 = SameMaxPool2d(kernel_size=7, stride=1)

        # Stem blocks.
        self._register("stem0_conv", nn.Conv2d(1, 8, kernel_size=5, padding=2, bias=True))
        self._register(
            "stem0_dilated_conv",
            nn.Conv2d(1, 8, kernel_size=5, padding=4, dilation=2, bias=True),
        )
        self._register("stem0_ln", ChannelLayerNorm(16))

        self._register("stem1_conv", nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=True))
        self._register(
            "stem1_dilated_conv",
            nn.Conv2d(1, 16, kernel_size=5, padding=4, dilation=2, bias=True),
        )
        self._register("stem1_ln", ChannelLayerNorm(32))

        # Context autoencoder (sep conv blocks).
        self._register("enc0_0_conv", SeparableConv2d(16, 16, kernel_size=5, padding=2))
        self._register("enc0_0_ln", ChannelLayerNorm(16))
        self._register("enc0_1_conv", SeparableConv2d(16, 32, kernel_size=5, padding=2))
        self._register("enc0_1_ln", ChannelLayerNorm(32))
        self._register("enc0_2_conv", SeparableConv2d(32, 64, kernel_size=5, padding=2))
        self._register("enc0_2_ln", ChannelLayerNorm(64))
        self._register("enc0_3_conv", SeparableConv2d(64, 128, kernel_size=5, padding=2))
        self._register("enc0_3_ln", ChannelLayerNorm(128))

        self._register("dec0_3_conv", SeparableConv2d(128, 128, kernel_size=5, padding=2))
        self._register("dec0_3_ln", ChannelLayerNorm(128))
        self._register("dec0_2_conv", SeparableConv2d(256, 64, kernel_size=5, padding=2))
        self._register("dec0_2_ln", ChannelLayerNorm(64))
        self._register("dec0_1_conv", SeparableConv2d(128, 32, kernel_size=5, padding=2))
        self._register("dec0_1_ln", ChannelLayerNorm(32))
        self._register("dec0_0_conv", SeparableConv2d(64, 16, kernel_size=5, padding=2))
        self._register("dec0_0_ln", ChannelLayerNorm(16))

        # Attention gate.
        self._register("attention_conv_d1", nn.Conv2d(32, 16, kernel_size=3, padding=1, dilation=1, bias=True))
        self._register("attention_conv_d3", nn.Conv2d(32, 16, kernel_size=3, padding=3, dilation=3, bias=True))
        self._register("attention_conv_d6", nn.Conv2d(32, 16, kernel_size=3, padding=6, dilation=6, bias=True))
        self._register("attention_s", nn.Conv2d(48, 32, kernel_size=1, padding=0, bias=True))

        # Refinement autoencoder (inverse bottleneck blocks).
        self._register_inv_block("enc1_0", in_ch=64, out_ch=32)
        self._register_inv_block("enc1_1", in_ch=32, out_ch=64)
        self._register_inv_block("enc1_2", in_ch=64, out_ch=128)
        self._register_inv_block("enc1_3", in_ch=128, out_ch=32)

        self._register_inv_block("dec1_3", in_ch=32, out_ch=27)
        self._register_inv_block("dec1_2", in_ch=59, out_ch=91)
        self._register_inv_block("dec1_1", in_ch=219, out_ch=42)
        self._register_inv_block("dec1_0", in_ch=106, out_ch=20)

        # Head blocks (three parallel branches).
        for k in range(3):
            prefix = f"head_{k}"
            self._register_inv_block(f"{prefix}_ibc", in_ch=52, out_ch=6)
            self._register(f"{prefix}_conv1", nn.Conv2d(6, 4, kernel_size=1, padding=0, bias=True))
            self._register(f"{prefix}_ln", ChannelLayerNorm(4))
            self._register(f"{prefix}_conv2", nn.Conv2d(4, 4, kernel_size=1, padding=0, bias=True))

        # Final prediction heads.
        self._register("head_conv_pos", nn.Conv2d(4, 1, kernel_size=5, padding=2, bias=True))
        self._register("head_conv_dir", nn.Conv2d(4, 2, kernel_size=5, padding=2, bias=True))
        self._register("head_conv_typ", nn.Conv2d(4, 1, kernel_size=5, padding=2, bias=True))
        self._register("nms_Gaussian_blur", nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False))
        self._init_nms_gaussian()

    def _register(self, name: str, module: nn.Module) -> None:
        setattr(self, name, module)
        self._keras_layer_map[name] = module

    def _register_inv_block(self, prefix: str, in_ch: int, out_ch: int) -> None:
        self._register(
            f"{prefix}_depthwise_conv",
            nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch, bias=False),
        )
        self._register(f"{prefix}_ln", ChannelLayerNorm(in_ch))
        self._register(f"{prefix}_conv_exp", nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, padding=0, bias=True))
        self._register(f"{prefix}_conv_red", nn.Conv2d(in_ch * 4, in_ch, kernel_size=1, padding=0, bias=True))
        self._register(f"{prefix}_conv_adj", nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True))

    def _init_nms_gaussian(self) -> None:
        k = np.exp(-((np.arange(5, dtype=np.float32) - 2.0) ** 2) / 2.0)
        k = k / np.sum(k)
        g = np.outer(k, k).astype(np.float32)
        self.nms_Gaussian_blur.weight.data.copy_(torch.from_numpy(g)[None, None, ...])

    def keras_layer_map(self) -> dict[str, nn.Module]:
        return self._keras_layer_map

    def _stem_block(self, prefix: str, x: torch.Tensor, max_pooling: bool) -> torch.Tensor:
        x1 = getattr(self, f"{prefix}_conv")(x)
        x2 = getattr(self, f"{prefix}_dilated_conv")(x)
        x = torch.cat([x1, x2], dim=1)
        x = getattr(self, f"{prefix}_ln")(x)
        x = F.gelu(x)
        x = self.maxpool2(x) if max_pooling else self.avgpool2(x)
        return x

    def _sep_block(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        x = getattr(self, f"{prefix}_conv")(x)
        x = getattr(self, f"{prefix}_ln")(x)
        return F.gelu(x)

    def _inv_block(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        x = getattr(self, f"{prefix}_depthwise_conv")(x)
        x = getattr(self, f"{prefix}_ln")(x)
        x = F.gelu(getattr(self, f"{prefix}_conv_exp")(x))
        x = getattr(self, f"{prefix}_conv_red")(x)
        x = x + x_input
        x = getattr(self, f"{prefix}_conv_adj")(x)
        return x

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        c1 = x.shape[1] // 2
        x1 = self.maxpool2(x[:, :c1])
        x2 = self.avgpool2(x[:, c1:])
        return torch.cat([x1, x2], dim=1)

    def _upsample(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return x

    def _head_block(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        x = self._inv_block(f"{prefix}_ibc", x)
        x = self._upsample(x)
        x = getattr(self, f"{prefix}_conv1")(x)
        x = getattr(self, f"{prefix}_ln")(x)
        x = F.gelu(getattr(self, f"{prefix}_conv2")(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected in [0, 255], shape [B,1,H,W].
        x = x.float()

        # Stem.
        s0 = self._stem_block("stem0", x, max_pooling=True)
        s1 = self._stem_block("stem1", x, max_pooling=False)

        # Context autoencoder.
        z = s0
        skip0: list[torch.Tensor] = []
        for i in range(4):
            z = self._sep_block(f"enc0_{i}", z)
            skip0.append(z)
            z = self._downsample(z)
        for i in (3, 2, 1, 0):
            z = self._sep_block(f"dec0_{i}", z)
            z = self._upsample(z, skip0[i])

        # Attention gate.
        d1 = F.gelu(self.attention_conv_d1(z))
        d3 = F.gelu(self.attention_conv_d3(z))
        d6 = F.gelu(self.attention_conv_d6(z))
        xd = torch.cat([d1, d3, d6], dim=1)
        xf = torch.sigmoid(self.attention_s(xd))
        z = z * xf
        z = torch.cat([z, s1], dim=1)

        # Refinement autoencoder.
        skip1: list[torch.Tensor] = []
        for i in range(4):
            z = self._inv_block(f"enc1_{i}", z)
            skip1.append(z)
            z = self._downsample(z)
        for i in (3, 2, 1, 0):
            z = self._inv_block(f"dec1_{i}", z)
            z = self._upsample(z, skip1[i])

        # Heads.
        h0 = self._head_block("head_0", z)
        h1 = self._head_block("head_1", z)
        h2 = self._head_block("head_2", z)

        pos = torch.sigmoid(self.head_conv_pos(h0))
        d = self.head_conv_dir(h1)
        direction = torch.atan2(d[:, 1:2], d[:, 0:1])
        typ = torch.sigmoid(self.head_conv_typ(h2))

        pos_gb = self.nms_Gaussian_blur(pos)
        pos_nms = self.maxpool7_stride1(pos_gb)
        pos_nms = pos_gb * (pos_gb == pos_nms).to(pos_gb.dtype)

        return torch.cat([pos, direction, typ, pos_nms], dim=1)

