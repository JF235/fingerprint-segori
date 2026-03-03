from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

# Force CPU for deterministic Keras/TensorFlow compatibility checks.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pyfing.pytorch.compat import compare_inference, force_tensorflow_cpu, transfer_keras_to_torch
from pyfing.pytorch.registry import get_model_spec, list_model_specs


def _resolve_specs(args: argparse.Namespace):
    if args.all:
        return list_model_specs()
    if not args.model:
        raise ValueError("Provide --all or --model <name[,name2,...]>")
    names = [n.strip() for n in args.model.split(",") if n.strip()]
    return [get_model_spec(n) for n in names]


def _state_diff_max(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[float, float]:
    max_abs = 0.0
    mean_abs = 0.0
    for k in a:
        d = (a[k].float() - b[k].float()).abs()
        max_abs = max(max_abs, float(d.max().item()))
        mean_abs = max(mean_abs, float(d.mean().item()))
    return max_abs, mean_abs


def _collect_first_images(images_dir: Path, limit: int = 10) -> list[Path]:
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: str(p.relative_to(images_dir)))
    return files[:limit]


def _build_real_fixture(images_dir: Path, model_name: str, channels: int) -> np.ndarray:
    image_paths = _collect_first_images(images_dir, limit=10)
    if not image_paths:
        raise ValueError(f"No image files found in {images_dir}")
    if len(image_paths) < 10:
        raise ValueError(f"Expected at least 10 images in {images_dir}, found {len(image_paths)}")

    batch: list[np.ndarray] = []
    for p in image_paths:
        img = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        img = cv.resize(img, (256, 256), interpolation=cv.INTER_AREA)
        base = img.astype(np.float32)
        if model_name in {"sufs", "leader"}:
            x = base[:, :, None]
        elif model_name == "snfoe":
            # SNFOE expects fingerprint + binary mask.
            mask = np.ones_like(base, dtype=np.float32)
            x = np.dstack([base, mask])
        elif model_name == "snffe":
            # SNFFE expects fingerprint + mask + quantized orientation [0..255].
            mask = np.ones_like(base, dtype=np.float32)
            orientation_q = np.full_like(base, 128.0, dtype=np.float32)
            x = np.dstack([base, mask, orientation_q])
        elif model_name == "snfen":
            # SNFEN expects fingerprint + mask + quantized orientation + quantized ridge period.
            mask = np.ones_like(base, dtype=np.float32)
            orientation_q = np.full_like(base, 128.0, dtype=np.float32)
            ridge_period_q = np.full_like(base, 100.0, dtype=np.float32)
            x = np.dstack([base, mask, orientation_q, ridge_period_q])
        else:
            x = np.repeat(base[:, :, None], channels, axis=2)

        if x.shape[-1] != channels:
            raise ValueError(
                f"Model {model_name} expected {channels} channels, built fixture with {x.shape[-1]}"
            )
        batch.append(x)
    return np.stack(batch, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Keras<->PyTorch compatibility for pyfing.pytorch converted models.",
    )
    parser.add_argument("--all", action="store_true", help="Validate all known models.")
    parser.add_argument("--model", type=str, default=None, help="Model name(s), comma separated.")
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Directory containing converted .pth files (default: pyfing/pytorch/models).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed used to generate inference fixtures.")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional directory with real images. Uses the first 10 files in lexicographic order.",
    )
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for inference validation.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for inference validation.")
    parser.add_argument("--max-abs-tol", type=float, default=5e-2, help="Max absolute error tolerance for compatibility pass/fail.")
    parser.add_argument("--mean-abs-tol", type=float, default=1e-3, help="Mean absolute error tolerance for compatibility pass/fail.")
    parser.add_argument("--report-json", type=str, default=None, help="Optional JSON output report path.")
    args = parser.parse_args()

    specs = _resolve_specs(args)
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else None
    if images_dir is not None and not images_dir.is_dir():
        raise ValueError(f"--images-dir is not a directory: {images_dir}")

    force_tensorflow_cpu()
    out_report: list[dict] = []
    failed = False

    for spec in specs:
        # Build expected reference from Keras weights.
        keras_model = spec.build_keras_model(spec.keras_weights)
        expected_model = spec.build_torch_model()
        transfer_keras_to_torch(keras_model, expected_model)
        expected_sd = {k: v.detach().cpu() for k, v in expected_model.state_dict().items()}

        # Load converted .pth.
        weights_path = Path(args.weights_dir) / spec.torch_weights.name if args.weights_dir else spec.torch_weights
        if not weights_path.exists():
            print(f"[{spec.name}] FAIL missing weights: {weights_path}")
            failed = True
            out_report.append({"model": spec.name, "passed": False, "error": f"missing {weights_path}"})
            continue
        loaded_model = spec.build_torch_model()
        loaded_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        loaded_model.eval()
        loaded_sd = {k: v.detach().cpu() for k, v in loaded_model.state_dict().items()}

        if set(expected_sd) != set(loaded_sd):
            print(f"[{spec.name}] FAIL state_dict keys mismatch")
            failed = True
            out_report.append({"model": spec.name, "passed": False, "error": "state_dict keys mismatch"})
            continue

        tensor_max_abs, tensor_max_mean = _state_diff_max(expected_sd, loaded_sd)
        tensors_ok = tensor_max_abs <= 1e-6

        # Inference comparison using loaded model.
        if images_dir is None:
            fixture = spec.fixture_input(args.seed)
        else:
            sample_shape = spec.fixture_input(args.seed).shape
            if len(sample_shape) != 4:
                raise ValueError(f"Unexpected fixture shape for model {spec.name}: {sample_shape}")
            fixture = _build_real_fixture(images_dir, model_name=spec.name, channels=sample_shape[-1])
        inf = compare_inference(
            keras_model=keras_model,
            torch_model=loaded_model,
            fixture_input=fixture,
            rtol=args.rtol,
            atol=args.atol,
            max_abs_tol=args.max_abs_tol,
            mean_abs_tol=args.mean_abs_tol,
            periodic_output_channels=spec.periodic_output_channels,
        )

        passed = tensors_ok and inf.passed
        failed = failed or (not passed)
        print(
            f"[{spec.name}] "
            f"weights_max_abs={tensor_max_abs:.3e} "
            f"inference_mean_abs={inf.mean_abs:.3e} "
            f"inference_max_abs={inf.max_abs:.3e} "
            f"allclose={inf.allclose} "
            f"passed={passed}"
        )
        out_report.append(
            {
                "model": spec.name,
                "weights": str(weights_path),
                "tensor_max_abs": tensor_max_abs,
                "tensor_max_mean": tensor_max_mean,
                "inference": {
                    "passed": inf.passed,
                    "max_abs": inf.max_abs,
                    "mean_abs": inf.mean_abs,
                    "allclose": inf.allclose,
                    "rtol": inf.rtol,
                    "atol": inf.atol,
                    "max_abs_tol": inf.max_abs_tol,
                    "mean_abs_tol": inf.mean_abs_tol,
                    "keras_shape": inf.keras_shape,
                    "torch_shape": inf.torch_shape,
                },
                "passed": passed,
            }
        )

    if args.report_json:
        p = Path(args.report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"models": out_report}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Report written to {p}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
