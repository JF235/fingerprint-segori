from __future__ import annotations

import numpy as np

from pyfing.definitions import Image, Minutia

from .algorithms import LeaderTorch, SnfenTorch, SnffeTorch, SnfoeTorch, SufsTorch

_sufs_alg = None
_snfoe_alg = None
_snffe_alg = None
_snfen_alg = None
_leader_alg = None


def fingerprint_segmentation(fingerprint: Image, dpi: int = 500, method: str = "SUFS") -> Image:
    global _sufs_alg
    if method != "SUFS":
        raise ValueError(f"Invalid method ({method}). PyTorch backend currently supports only SUFS for segmentation.")
    if _sufs_alg is None:
        _sufs_alg = SufsTorch()
    _sufs_alg.parameters.image_dpi = dpi
    return _sufs_alg.run(fingerprint)


def orientation_field_estimation(
    fingerprint: Image,
    segmentation_mask: Image | None = None,
    dpi: int = 500,
    method: str = "SNFOE",
) -> np.ndarray:
    global _snfoe_alg
    if method != "SNFOE":
        raise ValueError(f"Invalid method ({method}). PyTorch backend currently supports only SNFOE for orientation.")
    if _snfoe_alg is None:
        _snfoe_alg = SnfoeTorch()
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return _snfoe_alg.run(fingerprint, segmentation_mask, dpi)[0]


def frequency_estimation(
    fingerprint: Image,
    orientation_field: np.ndarray,
    segmentation_mask: Image | None = None,
    dpi: int = 500,
    method: str = "SNFFE",
) -> np.ndarray:
    global _snffe_alg
    if method != "SNFFE":
        raise ValueError(f"Invalid method ({method}). PyTorch backend currently supports only SNFFE for frequency.")
    if _snffe_alg is None:
        _snffe_alg = SnffeTorch()
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return _snffe_alg.run(fingerprint, segmentation_mask, orientation_field, dpi)


def fingerprint_enhancement(
    fingerprint: Image,
    orientation_field: np.ndarray,
    ridge_period_map: np.ndarray,
    segmentation_mask: Image | None = None,
    dpi: int = 500,
    method: str = "SNFEN",
) -> Image:
    global _snfen_alg
    if method != "SNFEN":
        raise ValueError(f"Invalid method ({method}). PyTorch backend currently supports only SNFEN for enhancement.")
    if _snfen_alg is None:
        _snfen_alg = SnfenTorch()
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return _snfen_alg.run(fingerprint, segmentation_mask, orientation_field, ridge_period_map, dpi)


def minutiae_extraction(fingerprint: Image, dpi: int = 500, method: str = "LEADER") -> list[Minutia]:
    global _leader_alg
    if method != "LEADER":
        raise ValueError(f"Invalid method ({method}).")
    if _leader_alg is None:
        _leader_alg = LeaderTorch()
    return _leader_alg.run(fingerprint, dpi)

