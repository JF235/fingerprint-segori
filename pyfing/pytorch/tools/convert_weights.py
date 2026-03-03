from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Force CPU for deterministic Keras/TensorFlow compatibility checks.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pyfing.pytorch.compat import convert_spec, sha256, write_manifest
from pyfing.pytorch.registry import get_model_spec, list_model_specs


def _resolve_specs(args: argparse.Namespace):
    if args.all:
        return list_model_specs()
    if not args.model:
        raise ValueError("Provide --all or --model <name[,name2,...]>")
    names = [n.strip() for n in args.model.split(",") if n.strip()]
    return [get_model_spec(n) for n in names]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert pyfing Keras .weights.h5 models to PyTorch .pth and optionally validate compatibility.",
    )
    parser.add_argument("--all", action="store_true", help="Convert all known models.")
    parser.add_argument("--model", type=str, default=None, help="Model name(s), comma separated.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .pth files. Default: pyfing/pytorch/models",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed used to generate compatibility fixtures.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for inference comparison.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for inference comparison.")
    parser.add_argument("--max-abs-tol", type=float, default=5e-2, help="Max absolute error tolerance for compatibility pass/fail.")
    parser.add_argument("--mean-abs-tol", type=float, default=1e-3, help="Mean absolute error tolerance for compatibility pass/fail.")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip inference compatibility checks (tensor mapping is still validated).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest JSON path. Default: <output-dir>/manifest.json",
    )
    args = parser.parse_args()

    specs = _resolve_specs(args)
    output_dir = Path(args.output_dir) if args.output_dir else specs[0].torch_weights.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failed = False
    for spec in specs:
        out_path = output_dir / spec.torch_weights.name
        res = convert_spec(
            spec=spec,
            output_path=out_path,
            verify_inference=not args.no_verify,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
            max_abs_tol=args.max_abs_tol,
            mean_abs_tol=args.mean_abs_tol,
        )
        res["sha256"] = sha256(out_path)
        res["size_bytes"] = out_path.stat().st_size
        results.append(res)
        failed = failed or (not bool(res["passed"]))
        print(
            f"[{spec.name}] saved={out_path} "
            f"tensor_max_abs={res['tensor_diffs']['max_abs']:.3e} "
            + (
                "inference=SKIPPED"
                if res["inference"] is None
                else (
                    f"inference_max_abs={res['inference']['max_abs']:.3e} "
                    f"inference_mean_abs={res['inference']['mean_abs']:.3e} "
                    f"allclose={res['inference']['allclose']} "
                    f"passed={res['inference']['passed']}"
                )
            )
        )

    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    manifest = {"tool": "pyfing.pytorch.tools.convert_weights", "models": results}
    write_manifest(manifest_path, manifest)
    print(f"Manifest written to {manifest_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
