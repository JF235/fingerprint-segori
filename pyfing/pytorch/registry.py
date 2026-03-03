from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch.nn as nn

from .leader_model import LeaderNet
from .snfen_model import SnfenNet
from .snffe_model import SnffeNet
from .snfoe_model import SnfoeNet
from .sufs_model import SufsNet


@dataclass(frozen=True)
class ModelSpec:
    name: str
    keras_weights: Path
    torch_weights: Path
    build_keras_model: Callable[[Path], object]
    build_torch_model: Callable[[], nn.Module]
    fixture_input: Callable[[int], np.ndarray]
    periodic_output_channels: dict[int, float] | None = None


_ROOT = Path(__file__).resolve().parent
_PYFING_ROOT = _ROOT.parent
_KERAS_MODEL_DIR = _PYFING_ROOT / "models"
_TORCH_MODEL_DIR = _ROOT / "models"


def _build_sufs_keras(weights_path: Path):
    from pyfing.segmentation import Sufs

    return Sufs(model_weights=str(weights_path)).model


def _build_snfoe_keras(weights_path: Path):
    from pyfing.orientations import Snfoe

    return Snfoe(model_weights=str(weights_path)).model


def _build_snffe_keras(weights_path: Path):
    from pyfing.frequencies import Snffe

    return Snffe(model_weights=str(weights_path)).model


def _build_snfen_keras(weights_path: Path):
    from pyfing.enhancement import Snfen

    return Snfen(model_weights=str(weights_path)).model


def _build_leader_keras(weights_path: Path):
    from pyfing.minutiae import Leader

    return Leader(model_weights=str(weights_path)).model


def _rng_input(seed: int, shape: tuple[int, int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8).astype(np.float32)


MODEL_SPECS: dict[str, ModelSpec] = {
    "sufs": ModelSpec(
        name="sufs",
        keras_weights=_KERAS_MODEL_DIR / "SUFS.weights.h5",
        torch_weights=_TORCH_MODEL_DIR / "SUFS.pth",
        build_keras_model=_build_sufs_keras,
        build_torch_model=SufsNet,
        fixture_input=lambda seed: _rng_input(seed, (1, 256, 256, 1)),
    ),
    "snfoe": ModelSpec(
        name="snfoe",
        keras_weights=_KERAS_MODEL_DIR / "SNFOE.weights.h5",
        torch_weights=_TORCH_MODEL_DIR / "SNFOE.pth",
        build_keras_model=_build_snfoe_keras,
        build_torch_model=SnfoeNet,
        fixture_input=lambda seed: _rng_input(seed, (1, 256, 256, 2)),
        periodic_output_channels={0: float(np.pi)},
    ),
    "snffe": ModelSpec(
        name="snffe",
        keras_weights=_KERAS_MODEL_DIR / "SNFFE.weights.h5",
        torch_weights=_TORCH_MODEL_DIR / "SNFFE.pth",
        build_keras_model=_build_snffe_keras,
        build_torch_model=SnffeNet,
        fixture_input=lambda seed: _rng_input(seed, (1, 256, 256, 3)),
    ),
    "snfen": ModelSpec(
        name="snfen",
        keras_weights=_KERAS_MODEL_DIR / "SNFEN.weights.h5",
        torch_weights=_TORCH_MODEL_DIR / "SNFEN.pth",
        build_keras_model=_build_snfen_keras,
        build_torch_model=SnfenNet,
        fixture_input=lambda seed: _rng_input(seed, (1, 256, 256, 4)),
    ),
    "leader": ModelSpec(
        name="leader",
        keras_weights=_KERAS_MODEL_DIR / "LEADER.weights.h5",
        torch_weights=_TORCH_MODEL_DIR / "LEADER.pth",
        build_keras_model=_build_leader_keras,
        build_torch_model=LeaderNet,
        fixture_input=lambda seed: _rng_input(seed, (1, 256, 256, 1)),
        periodic_output_channels={1: float(2 * np.pi)},
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    key = name.strip().lower()
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_SPECS))}")
    return MODEL_SPECS[key]


def list_model_specs() -> list[ModelSpec]:
    return [MODEL_SPECS[k] for k in sorted(MODEL_SPECS)]
