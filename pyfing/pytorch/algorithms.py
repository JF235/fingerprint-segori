from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from pyfing.definitions import Image, Minutia
from pyfing.enhancement import EnhancementAlgorithm, SnfenParameters
from pyfing.frequencies import FrequencyEstimationAlgorithm, SnffeParameters
from pyfing.minutiae import EndToEndMinutiaExtractionAlgorithm, LeaderParameters
from pyfing.orientations import OrientationEstimationAlgorithm, SnfoeParameters
from pyfing.segmentation import SegmentationAlgorithm, SufsParameters

from .leader_model import LeaderNet
from .registry import get_model_spec
from .snfen_model import SnfenNet
from .snffe_model import SnffeNet
from .snfoe_model import SnfoeNet
from .sufs_model import SufsNet


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_state(model: torch.nn.Module, weights_path: str | Path, device: str) -> torch.nn.Module:
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _predict_nhwc(model: torch.nn.Module, arr: np.ndarray, device: str) -> np.ndarray:
    # arr: [N,H,W,C] float32
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        y = model(x).detach().cpu().numpy()
    if y.ndim == 4:
        y = np.transpose(y, (0, 2, 3, 1))
    return y


class SufsTorch(SegmentationAlgorithm):
    def __init__(
        self,
        parameters: SufsParameters | None = None,
        model_weights: str | None = None,
        model: SufsNet | None = None,
        device: str = DEFAULT_DEVICE,
    ):
        if parameters is None:
            parameters = SufsParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
        else:
            if model_weights is None:
                model_weights = str(get_model_spec("sufs").torch_weights)
            self.model = _load_state(SufsNet(), model_weights, device)

    def _compute_size(self, original_w, original_h):
        w, h = original_w, original_h
        if self.parameters.dnn_input_dpi != self.parameters.image_dpi:
            f = self.parameters.dnn_input_dpi / self.parameters.image_dpi
            w, h = int(round(original_w * f)), int(round(original_h * f))
        size_mult = self.parameters.dnn_input_size_multiple
        input_w = (w + self.parameters.border + size_mult - 1) // size_mult * size_mult
        input_h = (h + self.parameters.border + size_mult - 1) // size_mult * size_mult
        border_left, border_top = (input_w - w) // 2, (input_h - h) // 2
        border_right, border_bottom = input_w - w - border_left, input_h - h - border_top
        return w, h, border_left, border_top, border_right, border_bottom

    def _adjust_input(self, image, w, h, border_left, border_top, border_right, border_bottom):
        original_h, original_w = image.shape
        if w != original_w or h != original_h:
            image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC)
        return cv.copyMakeBorder(
            image,
            border_top,
            border_bottom,
            border_left,
            border_right,
            cv.BORDER_CONSTANT,
            value=image[0, 0].tolist(),
        )

    def _adjust_output(self, mask, w, h, border_left, border_top, border_right, border_bottom, original_w, original_h):
        mask = mask[border_top : border_top + h, border_left : border_left + w]
        if w != original_w or h != original_h:
            mask = cv.resize(mask, (original_w, original_h), interpolation=cv.INTER_NEAREST)
        return mask

    def run(self, image: Image, intermediate_results=None) -> Image:
        original_h, original_w = image.shape
        size_info = self._compute_size(original_w, original_h)
        ir = self._adjust_input(image, *size_info)
        preds = _predict_nhwc(self.model, ir[np.newaxis, ..., np.newaxis].astype(np.float32), self.device)
        mask = np.where(preds[0, ..., 0] < self.parameters.threshold, 0, 255).astype(np.uint8)
        return self._adjust_output(mask, *size_info, original_w, original_h)

    def run_on_db(self, images: list[Image]) -> list[Image]:
        return [self.run(img) for img in images]


class SnfoeTorch(OrientationEstimationAlgorithm):
    def __init__(
        self,
        parameters: SnfoeParameters | None = None,
        model_weights: str | None = None,
        model: SnfoeNet | None = None,
        device: str = DEFAULT_DEVICE,
    ):
        if parameters is None:
            parameters = SnfoeParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
        else:
            if model_weights is None:
                model_weights = str(get_model_spec("snfoe").torch_weights)
            self.model = _load_state(SnfoeNet(), model_weights, device)

    def run(self, image: Image, mask: Image = None, dpi: int = 500, intermediate_results=None) -> tuple[np.ndarray, np.ndarray]:
        if mask is None:
            mask = np.full_like(image, 255)
        p = self.parameters
        original_h, original_w = image.shape

        if dpi != p.dnn_input_dpi:
            f = p.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            mask = cv.resize(mask, None, fx=f, fy=f, interpolation=cv.INTER_NEAREST)

        h, w = image.shape
        size_mult = p.dnn_input_size_multiple
        input_w, input_h = (w + size_mult - 1) // size_mult * size_mult, (h + size_mult - 1) // size_mult * size_mult
        border_left, border_top = (input_w - w) // 2, (input_h - h) // 2
        border_right, border_bottom = input_w - w - border_left, input_h - h - border_top

        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value=image[0, 0].tolist())
        mr = cv.copyMakeBorder(mask // 255, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        x = np.dstack((ir, mr))[np.newaxis, ...].astype(np.float32)
        orientations = _predict_nhwc(self.model, x, self.device)[0]
        orientations = orientations[border_top : border_top + h, border_left : border_left + w]

        if dpi != p.dnn_input_dpi:
            cos2, sin2 = np.cos(orientations * 2), np.sin(orientations * 2)
            cos2 = cv.resize(cos2, (original_w, original_h), interpolation=cv.INTER_CUBIC)
            sin2 = cv.resize(sin2, (original_w, original_h), interpolation=cv.INTER_CUBIC)
            orientations = np.arctan2(sin2, cos2) / 2

        strengths = np.ones_like(orientations)
        return orientations, strengths


class SnffeTorch(FrequencyEstimationAlgorithm):
    def __init__(
        self,
        parameters: SnffeParameters | None = None,
        model_weights: str | None = None,
        model: SnffeNet | None = None,
        device: str = DEFAULT_DEVICE,
    ):
        if parameters is None:
            parameters = SnffeParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
        else:
            if model_weights is None:
                model_weights = str(get_model_spec("snffe").torch_weights)
            self.model = _load_state(SnffeNet(), model_weights, device)

    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, dpi: int = 500, intermediate_results=None) -> np.ndarray:
        p = self.parameters
        original_h, original_w = image.shape

        if dpi != p.dnn_input_dpi:
            f = p.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            mask = cv.resize(mask, None, fx=f, fy=f, interpolation=cv.INTER_NEAREST)
            cos2, sin2 = np.cos(orientation_field * 2), np.sin(orientation_field * 2)
            cos2 = cv.resize(cos2, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            sin2 = cv.resize(sin2, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            orientation_field = np.arctan2(sin2, cos2) / 2

        h, w = image.shape
        size_mult = p.dnn_input_size_multiple
        input_w, input_h = (w + size_mult - 1) // size_mult * size_mult, (h + size_mult - 1) // size_mult * size_mult
        border_left, border_top = (input_w - w) // 2, (input_h - h) // 2
        border_right, border_bottom = input_w - w - border_left, input_h - h - border_top

        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value=image[0, 0].tolist())
        mr = cv.copyMakeBorder(mask // 255, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = cv.copyMakeBorder(orientation_field, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = np.round((orr % np.pi) * 255 / np.pi).clip(0, 255).astype(np.uint8)

        x = np.dstack((ir, mr, orr))[np.newaxis, ...].astype(np.float32)
        rp = _predict_nhwc(self.model, x, self.device)[0]
        rp = rp[border_top : border_top + h, border_left : border_left + w]

        if dpi != p.dnn_input_dpi:
            rp = cv.resize(rp, (original_w, original_h), interpolation=cv.INTER_CUBIC)
            rp *= dpi / p.dnn_input_dpi
        rp /= 10.0
        return rp


class SnfenTorch(EnhancementAlgorithm):
    def __init__(
        self,
        parameters: SnfenParameters | None = None,
        model_weights: str | None = None,
        model: SnfenNet | None = None,
        device: str = DEFAULT_DEVICE,
    ):
        if parameters is None:
            parameters = SnfenParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
        else:
            if model_weights is None:
                model_weights = str(get_model_spec("snfen").torch_weights)
            self.model = _load_state(SnfenNet(), model_weights, device)

    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, ridge_periods: np.ndarray, dpi: int = 500, intermediate_results=None) -> Image:
        p = self.parameters
        original_h, original_w = image.shape

        if dpi != p.dnn_input_dpi:
            f = p.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            mask = cv.resize(mask, None, fx=f, fy=f, interpolation=cv.INTER_NEAREST)
            cos2, sin2 = np.cos(orientation_field * 2), np.sin(orientation_field * 2)
            cos2 = cv.resize(cos2, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            sin2 = cv.resize(sin2, None, fx=f, fy=f, interpolation=cv.INTER_CUBIC)
            orientation_field = np.arctan2(sin2, cos2) / 2
            ridge_periods = cv.resize(ridge_periods, (original_w, original_h), interpolation=cv.INTER_CUBIC)
            ridge_periods *= f

        h, w = image.shape
        size_mult = p.dnn_input_size_multiple
        input_w, input_h = (w + size_mult - 1) // size_mult * size_mult, (h + size_mult - 1) // size_mult * size_mult
        border_left, border_top = (input_w - w) // 2, (input_h - h) // 2
        border_right, border_bottom = input_w - w - border_left, input_h - h - border_top

        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value=image[0, 0].tolist())
        mr = cv.copyMakeBorder(mask // 255, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = cv.copyMakeBorder(orientation_field, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = np.round((orr % np.pi) * 255 / np.pi).clip(0, 255).astype(np.uint8)
        rpr = cv.copyMakeBorder(ridge_periods, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        rpr = np.round(rpr * 10).clip(0, 255).astype(np.uint8)

        x = np.dstack((ir, mr, orr, rpr))[np.newaxis, ...].astype(np.float32)
        en = _predict_nhwc(self.model, x, self.device)[0]
        en = en[border_top : border_top + h, border_left : border_left + w]
        en = np.clip(np.round(en * 255), 0, 255).astype(np.uint8)

        if dpi != p.dnn_input_dpi:
            en = cv.resize(en, (original_w, original_h), interpolation=cv.INTER_CUBIC)
        return en

    def run_on_db(
        self,
        images: list[Image],
        masks: list[Image],
        orientation_fields: list[np.ndarray],
        ridge_periods: list[np.ndarray],
        dpi_of_images=None,
    ) -> list[Image]:
        dpi_list = [self.parameters.dnn_input_dpi] * len(images) if dpi_of_images is None else dpi_of_images
        return [self.run(i, m, o, r, d) for i, m, o, r, d in zip(images, masks, orientation_fields, ridge_periods, dpi_list)]


class LeaderTorch(EndToEndMinutiaExtractionAlgorithm):
    def __init__(
        self,
        parameters: LeaderParameters | None = None,
        model_weights: str | None = None,
        model: LeaderNet | None = None,
        device: str = DEFAULT_DEVICE,
    ):
        if parameters is None:
            parameters = LeaderParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
        else:
            if model_weights is None:
                model_weights = str(get_model_spec("leader").torch_weights)
            self.model = _load_state(LeaderNet(), model_weights, device)

    def _compute_single_size(self, x, size_mult, min_border):
        input_x = (x + size_mult - 1) // size_mult * size_mult
        border_start = (input_x - x) // 2
        border_end = input_x - x - border_start
        add_x = max(0, min_border - border_start) + max(0, min_border - border_end) + border_start + border_end
        return (x + add_x + size_mult - 1) // size_mult * size_mult

    def _compute_size(self, w, h, size_mult, min_border):
        return self._compute_single_size(w, size_mult, min_border), self._compute_single_size(h, size_mult, min_border)

    def _get_minutiae(self, out: np.ndarray) -> list[Minutia]:
        coords = np.argwhere(out[..., 3] >= self.parameters.minutia_quality_threshold).tolist()
        return [
            Minutia(
                int(ix),
                int(iy),
                float(out[iy, ix, 1]),
                "E" if float(out[iy, ix, 2]) >= self.parameters.type_threshold else "B",
                float(out[iy, ix, 3]),
            )
            for iy, ix in coords
        ]

    def run(self, image: Image, dpi: int = 500, intermediate_results: list | None = None) -> list[Minutia]:
        p = self.parameters
        if dpi != p.dnn_input_dpi:
            dpi_scale = p.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx=dpi_scale, fy=dpi_scale, interpolation=cv.INTER_CUBIC)

        h, w = image.shape
        size_mult = p.dnn_input_size_multiple
        input_w, input_h = self._compute_size(w, h, size_mult, 0)
        border_left, border_top = (input_w - w) // 2, (input_h - h) // 2
        border_right, border_bottom = input_w - w - border_left, input_h - h - border_top
        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value=image[0, 0].tolist())

        y = _predict_nhwc(self.model, ir[np.newaxis, ..., np.newaxis].astype(np.float32), self.device)[0]
        y = y[border_top : border_top + h, border_left : border_left + w]
        minutiae = self._get_minutiae(y)
        if dpi != p.dnn_input_dpi:
            scale = dpi / p.dnn_input_dpi
            minutiae = [Minutia(int(round(m.x * scale)), int(round(m.y * scale)), m.direction, m.type, m.quality) for m in minutiae]
        return minutiae

    def run_on_db(self, images: list[Image], dpi_of_images=None) -> list[list[Minutia]]:
        dpi_list = [self.parameters.dnn_input_dpi] * len(images) if dpi_of_images is None else dpi_of_images
        return [self.run(img, dpi) for img, dpi in zip(images, dpi_list)]

