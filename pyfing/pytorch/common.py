from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding_2d(h: int, w: int, kh: int, kw: int, sh: int, sw: int) -> tuple[int, int, int, int]:
    out_h = math.ceil(h / sh)
    out_w = math.ceil(w / sw)
    pad_h = max((out_h - 1) * sh + kh - h, 0)
    pad_w = max((out_w - 1) * sw + kw - w, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom


class SameMaxPool2d(nn.Module):
    """TensorFlow/Keras-like MaxPool2D with padding='same'."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        left, right, top, bottom = _same_padding_2d(
            h, w, self.kernel_size, self.kernel_size, self.stride, self.stride
        )
        pad_val = torch.finfo(x.dtype).min
        x = F.pad(x, (left, right, top, bottom), value=pad_val)
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)


class SameAvgPool2d(nn.Module):
    """TensorFlow/Keras-like AvgPool2D with padding='same'."""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        left, right, top, bottom = _same_padding_2d(
            h, w, self.kernel_size, self.kernel_size, self.stride, self.stride
        )
        x = F.pad(x, (left, right, top, bottom), value=0.0)
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            count_include_pad=True,
        )


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel axis for NCHW tensors (Keras LayerNormalization axis=-1)."""

    def __init__(self, channels: int, eps: float = 1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class SeparableConv2d(nn.Module):
    """Keras-like SeparableConv2D (depthwise + pointwise with shared bias on pointwise)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, conv_name: str, bn_name: str):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99)
        self.relu = nn.ReLU(inplace=False)
        self._keras_names = (conv_name, bn_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


def conv2d_kernel_to_torch(kernel: np.ndarray) -> torch.Tensor:
    # Keras Conv2D: (kh, kw, in, out) -> PyTorch: (out, in, kh, kw)
    return torch.from_numpy(kernel).permute(3, 2, 0, 1).contiguous()


def depthwise_kernel_to_torch(kernel: np.ndarray) -> torch.Tensor:
    # Keras depthwise: (kh, kw, in, depth_mult) -> PyTorch grouped:
    # (out=in*depth_mult, 1, kh, kw)
    kh, kw, in_ch, dmult = kernel.shape
    out = np.transpose(kernel, (2, 3, 0, 1)).reshape(in_ch * dmult, 1, kh, kw)
    return torch.from_numpy(out).contiguous()


def assign_keras_weights_to_torch_layer(keras_weights: list[np.ndarray], torch_layer: nn.Module) -> None:
    if isinstance(torch_layer, SeparableConv2d):
        if len(keras_weights) != 3:
            raise ValueError("SeparableConv2D expects 3 tensors: depthwise, pointwise, bias")
        torch_layer.depthwise.weight.data.copy_(depthwise_kernel_to_torch(keras_weights[0]))
        torch_layer.pointwise.weight.data.copy_(conv2d_kernel_to_torch(keras_weights[1]))
        torch_layer.pointwise.bias.data.copy_(torch.from_numpy(keras_weights[2]))
        return

    if isinstance(torch_layer, nn.Conv2d):
        weight = keras_weights[0]
        if torch_layer.groups == torch_layer.in_channels and weight.ndim == 4 and weight.shape[2] == torch_layer.in_channels:
            torch_layer.weight.data.copy_(depthwise_kernel_to_torch(weight))
        else:
            torch_layer.weight.data.copy_(conv2d_kernel_to_torch(weight))
        if len(keras_weights) == 2 and torch_layer.bias is not None:
            torch_layer.bias.data.copy_(torch.from_numpy(keras_weights[1]))
        return

    if isinstance(torch_layer, nn.BatchNorm2d):
        if len(keras_weights) != 4:
            raise ValueError("BatchNormalization expects 4 tensors: gamma, beta, mean, var")
        gamma, beta, moving_mean, moving_var = keras_weights
        torch_layer.weight.data.copy_(torch.from_numpy(gamma))
        torch_layer.bias.data.copy_(torch.from_numpy(beta))
        torch_layer.running_mean.data.copy_(torch.from_numpy(moving_mean))
        torch_layer.running_var.data.copy_(torch.from_numpy(moving_var))
        torch_layer.num_batches_tracked.data.zero_()
        return

    if isinstance(torch_layer, ChannelLayerNorm):
        if len(keras_weights) != 2:
            raise ValueError("LayerNormalization expects 2 tensors: gamma, beta")
        gamma, beta = keras_weights
        torch_layer.norm.weight.data.copy_(torch.from_numpy(gamma))
        torch_layer.norm.bias.data.copy_(torch.from_numpy(beta))
        return

    raise TypeError(f"Unsupported torch layer for conversion: {type(torch_layer).__name__}")


@dataclass
class TensorDiff:
    name: str
    max_abs: float
    mean_abs: float


def compare_tensors(a: torch.Tensor, b: torch.Tensor, name: str) -> TensorDiff:
    d = (a - b).abs()
    return TensorDiff(name=name, max_abs=float(d.max().item()), mean_abs=float(d.mean().item()))

