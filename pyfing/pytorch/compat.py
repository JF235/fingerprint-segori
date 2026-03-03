from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .common import (
    ChannelLayerNorm,
    SeparableConv2d,
    TensorDiff,
    assign_keras_weights_to_torch_layer,
    compare_tensors,
    conv2d_kernel_to_torch,
    depthwise_kernel_to_torch,
)
from .registry import ModelSpec


@dataclass
class InferenceReport:
    passed: bool
    allclose: bool
    max_abs: float
    mean_abs: float
    rtol: float
    atol: float
    max_abs_tol: float
    mean_abs_tol: float
    keras_shape: list[int]
    torch_shape: list[int]


def force_tensorflow_cpu() -> None:
    """Best-effort TensorFlow GPU disable for deterministic Keras comparisons."""
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _expected_tensors(keras_weights: list[np.ndarray], torch_layer: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if isinstance(torch_layer, SeparableConv2d):
        out["depthwise.weight"] = depthwise_kernel_to_torch(keras_weights[0])
        out["pointwise.weight"] = conv2d_kernel_to_torch(keras_weights[1])
        out["pointwise.bias"] = torch.from_numpy(keras_weights[2])
        return out

    if isinstance(torch_layer, nn.Conv2d):
        weight = keras_weights[0]
        if (
            torch_layer.groups == torch_layer.in_channels
            and weight.ndim == 4
            and weight.shape[2] == torch_layer.in_channels
        ):
            out["weight"] = depthwise_kernel_to_torch(weight)
        else:
            out["weight"] = conv2d_kernel_to_torch(weight)
        if len(keras_weights) > 1 and torch_layer.bias is not None:
            out["bias"] = torch.from_numpy(keras_weights[1])
        return out

    if isinstance(torch_layer, nn.BatchNorm2d):
        gamma, beta, moving_mean, moving_var = keras_weights
        out["weight"] = torch.from_numpy(gamma)
        out["bias"] = torch.from_numpy(beta)
        out["running_mean"] = torch.from_numpy(moving_mean)
        out["running_var"] = torch.from_numpy(moving_var)
        return out

    if isinstance(torch_layer, ChannelLayerNorm):
        gamma, beta = keras_weights
        out["norm.weight"] = torch.from_numpy(gamma)
        out["norm.bias"] = torch.from_numpy(beta)
        return out

    raise TypeError(f"Unsupported layer type for expected tensor extraction: {type(torch_layer).__name__}")


def _actual_tensors(torch_layer: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(torch_layer, SeparableConv2d):
        return {
            "depthwise.weight": torch_layer.depthwise.weight.detach().cpu(),
            "pointwise.weight": torch_layer.pointwise.weight.detach().cpu(),
            "pointwise.bias": torch_layer.pointwise.bias.detach().cpu(),
        }

    if isinstance(torch_layer, nn.Conv2d):
        out = {"weight": torch_layer.weight.detach().cpu()}
        if torch_layer.bias is not None:
            out["bias"] = torch_layer.bias.detach().cpu()
        return out

    if isinstance(torch_layer, nn.BatchNorm2d):
        return {
            "weight": torch_layer.weight.detach().cpu(),
            "bias": torch_layer.bias.detach().cpu(),
            "running_mean": torch_layer.running_mean.detach().cpu(),
            "running_var": torch_layer.running_var.detach().cpu(),
        }

    if isinstance(torch_layer, ChannelLayerNorm):
        return {
            "norm.weight": torch_layer.norm.weight.detach().cpu(),
            "norm.bias": torch_layer.norm.bias.detach().cpu(),
        }

    raise TypeError(f"Unsupported layer type for tensor extraction: {type(torch_layer).__name__}")


def transfer_keras_to_torch(keras_model: object, torch_model: nn.Module) -> list[TensorDiff]:
    if not hasattr(torch_model, "keras_layer_map"):
        raise TypeError("Torch model must implement keras_layer_map()")
    layer_map = torch_model.keras_layer_map()  # type: ignore[attr-defined]

    diffs: list[TensorDiff] = []
    missing: list[str] = []
    for layer in keras_model.layers:
        if not layer.weights:
            continue
        name = layer.name
        if name not in layer_map:
            missing.append(name)
            continue
        target = layer_map[name]
        keras_weights = [np.asarray(w) for w in layer.get_weights()]
        assign_keras_weights_to_torch_layer(keras_weights, target)

        expected = _expected_tensors(keras_weights, target)
        actual = _actual_tensors(target)
        for key in expected:
            diffs.append(compare_tensors(actual[key], expected[key], f"{name}.{key}"))

    if missing:
        raise KeyError(f"Torch model is missing mapped layers: {', '.join(missing)}")
    return diffs


def _torch_output_to_keras_layout(y: np.ndarray) -> np.ndarray:
    if y.ndim == 4:
        # NCHW -> NHWC
        return np.transpose(y, (0, 2, 3, 1))
    return y


def compare_inference(
    keras_model: object,
    torch_model: nn.Module,
    fixture_input: np.ndarray,
    rtol: float,
    atol: float,
    max_abs_tol: float,
    mean_abs_tol: float,
    periodic_output_channels: dict[int, float] | None = None,
) -> InferenceReport:
    keras_out = keras_model(fixture_input, training=False).numpy()

    x_torch = torch.from_numpy(fixture_input).permute(0, 3, 1, 2).float()
    torch_model.eval()
    with torch.no_grad():
        y_torch = torch_model(x_torch).detach().cpu().numpy()
    y_torch = _torch_output_to_keras_layout(y_torch)

    if keras_out.shape != y_torch.shape:
        raise ValueError(
            f"Output shape mismatch: keras={keras_out.shape}, torch={y_torch.shape}"
        )

    delta = np.abs(keras_out - y_torch)
    if periodic_output_channels:
        if delta.ndim < 1:
            raise ValueError("Invalid output shape for periodic channel handling.")
        for channel, period in periodic_output_channels.items():
            if channel < 0 or channel >= delta.shape[-1]:
                raise ValueError(
                    f"Invalid periodic output channel index {channel} for output shape {delta.shape}"
                )
            if period <= 0:
                raise ValueError(f"Invalid period for channel {channel}: {period}")
            dc = delta[..., channel]
            delta[..., channel] = np.minimum(dc, period - dc)

    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    if periodic_output_channels:
        tol = atol + rtol * np.abs(y_torch)
        allclose_ok = bool(np.all(delta <= tol))
    else:
        allclose_ok = bool(np.allclose(keras_out, y_torch, rtol=rtol, atol=atol))
    passed = bool(max_abs <= max_abs_tol and mean_abs <= mean_abs_tol)
    return InferenceReport(
        passed=passed,
        allclose=allclose_ok,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rtol=rtol,
        atol=atol,
        max_abs_tol=max_abs_tol,
        mean_abs_tol=mean_abs_tol,
        keras_shape=list(keras_out.shape),
        torch_shape=list(y_torch.shape),
    )


def convert_spec(
    spec: ModelSpec,
    output_path: Path | None = None,
    verify_inference: bool = True,
    seed: int = 1234,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    max_abs_tol: float = 5e-2,
    mean_abs_tol: float = 1e-3,
) -> dict[str, Any]:
    force_tensorflow_cpu()
    keras_model = spec.build_keras_model(spec.keras_weights)
    torch_model = spec.build_torch_model()

    diffs = transfer_keras_to_torch(keras_model, torch_model)
    max_tensor_abs = max((d.max_abs for d in diffs), default=0.0)
    max_tensor_mean = max((d.mean_abs for d in diffs), default=0.0)

    out = output_path if output_path is not None else spec.torch_weights
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model.state_dict(), out)

    inference = None
    if verify_inference:
        fixture = spec.fixture_input(seed)
        inference = compare_inference(
            keras_model=keras_model,
            torch_model=torch_model,
            fixture_input=fixture,
            rtol=rtol,
            atol=atol,
            max_abs_tol=max_abs_tol,
            mean_abs_tol=mean_abs_tol,
            periodic_output_channels=spec.periodic_output_channels,
        )

    return {
        "model": spec.name,
        "keras_weights": str(spec.keras_weights),
        "torch_weights": str(out),
        "tensor_diffs": {
            "count": len(diffs),
            "max_abs": max_tensor_abs,
            "max_mean": max_tensor_mean,
        },
        "inference": None if inference is None else asdict(inference),
        "passed": (inference.passed if inference is not None else True),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
