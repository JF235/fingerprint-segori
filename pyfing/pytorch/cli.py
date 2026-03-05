"""
CLI entry point for pyfing LEADER batch inference.

Usage:
    leader infer <input> <output> [options]
    leader infer --help

CUDA_VISIBLE_DEVICES is set *before* any torch import so the CUDA runtime
picks it up at initialisation time.
"""

import argparse
import json
import sys


# ─────────────────────────────────────────────────────────────────────────────
# GPU parsing helpers (same contract as fingernet/minutiaenet)
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpus(gpus_str: str) -> int | list[int]:
    """Parse GPU specification string.

    Examples::
        "0"       → 0          (CPU)
        "1"       → 1          (single GPU)
        "4"       → 4          (4 GPUs: 0,1,2,3)
        "[0,1]"   → [0, 1]     (specific physical GPUs)
    """
    if gpus_str.lower() == "none" or gpus_str == "0":
        return 0

    # Try plain integer
    try:
        n = int(gpus_str)
        if n < 0:
            raise ValueError("GPU count must be >= 0")
        return n
    except ValueError:
        pass

    # Try JSON list
    try:
        parsed = json.loads(gpus_str)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            if not parsed:
                raise ValueError("GPU list cannot be empty")
            if any(x < 0 for x in parsed):
                raise ValueError("GPU IDs must be non-negative")
            if len(set(parsed)) != len(parsed):
                raise ValueError("GPU list must not contain duplicates")
            return parsed
        raise ValueError("GPU list must contain only integers")
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Invalid GPU specification: {gpus_str!r}")


# Removed _to_cuda_visible_devices and _early_set_cuda_visible_devices
# to avoid relying on CUDA_VISIBLE_DEVICES env var, which doesn't work
# if torch is already imported. We now use torch.cuda.set_device() directly.


# ─────────────────────────────────────────────────────────────────────────────
# Command implementations
# ─────────────────────────────────────────────────────────────────────────────

def infer_command(args):
    """Run LEADER batch inference."""
    from .api import run_inference

    gpus = parse_gpus(args.gpus)

    print(f"\n{'='*70}")
    print("pyfing LEADER — Batch Minutiae Extraction")
    print(f"{'='*70}")
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"GPUs:         {gpus}")
    print(f"Batch size:   {args.batch_size} per GPU")
    print(f"DL workers:   {args.cores} per GPU")
    print(f"CPU workers:  {args.cpu_workers}")
    print(f"DPI:          {args.dpi}")
    print(f"Threshold:    {args.threshold}")
    print(f"Strategy:     {args.strategy}")
    print(f"Recursive:    {args.recursive}")
    print(f"Compile:      {args.compile}")
    print(f"Max dim:      {args.max_dim}")
    if args.weights:
        print(f"Weights:      {args.weights}")
    print(f"{'='*70}\n")

    run_inference(
        input_path=args.input,
        output_path=args.output,
        weights_path=args.weights,
        gpus=gpus,
        batch_size=args.batch_size,
        num_workers=args.cores,
        dpi=args.dpi,
        threshold=args.threshold,
        type_threshold=args.type_threshold,
        recursive=args.recursive,
        strategy=args.strategy,
        num_cpu_workers=args.cpu_workers,
        compile_model=args.compile,
        max_image_dim=args.max_dim,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="leader",
        description="pyfing LEADER — Lightweight End-to-end Attention-gated Dual autoencodER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU (default)
  leader infer images/ output/ --gpus 1 -b 8

  # 4-GPU DDP
  leader infer images/ output/ --gpus 4 -b 16 --recursive

  # Specific GPUs
  leader infer images/ output/ --gpus [2,3] -b 8

  # Non-standard DPI (e.g. 1000 dpi scanner)
  leader infer images/ output/ --gpus 1 --dpi 1000

  # Lower threshold for more minutiae
  leader infer images/ output/ --gpus 1 --threshold 0.4
        """,
    )

    subcommand_names = ("infer", "forward")

    def _add_infer_args(sp):
        sp.add_argument("input",  type=str, help="Input: image file, directory, or .txt/.list of paths")
        sp.add_argument("output", type=str, help="Output directory (minutiae/ subdir created automatically)")
        sp.add_argument(
            "--gpus", type=str, default="1",
            help='GPU config: "0"=CPU, "1"=single, "4"=4-GPU DDP, "[0,1]"=specific (default: 1)',
        )
        sp.add_argument(
            "-b", "--batch-size", dest="batch_size", type=int, default=8,
            help="Images per GPU per step (default: 8)",
        )
        sp.add_argument(
            "--cores", type=int, default=4,
            help="DataLoader worker processes per GPU (default: 4)",
        )
        sp.add_argument(
            "--cpu-workers", dest="cpu_workers", type=int, default=4,
            help="CPU threads for saving / post-processing (default: 4)",
        )
        sp.add_argument(
            "--dpi", type=int, default=500,
            help="Input image DPI; images are pre-scaled to 500 dpi before inference (default: 500)",
        )
        sp.add_argument(
            "--threshold", type=float, default=0.6,
            help="Minutia quality threshold 0–1 (default: 0.6)",
        )
        sp.add_argument(
            "--type-threshold", dest="type_threshold", type=float, default=0.5,
            help="Ending/Bifurcation decision threshold (default: 0.5)",
        )
        sp.add_argument(
            "--strategy", type=str, default="full_gpu",
            choices=["full_gpu", "hybrid"],
            help="Execution strategy: full_gpu (default) or hybrid",
        )
        sp.add_argument(
            "-r", "--recursive", action="store_true", default=True,
            help="Search directories recursively (default: on)",
        )
        sp.add_argument(
            "--compile", action="store_true",
            help="Compile model with torch.compile (experimental)",
        )
        sp.add_argument(
            "--max-dim", dest="max_dim", type=int, default=1024,
            help="Maximum image dimension after DPI scaling (default: 1024)",
        )
        sp.add_argument(
            "--weights", type=str, default=None,
            help="Path to custom LEADER .pth weights (default: bundled model)",
        )

    # Expanded help when no subcommand is given
    if any(h in sys.argv for h in ("-h", "--help")) and not any(
        c in sys.argv for c in subcommand_names
    ):
        sub_temp = parser.add_subparsers(dest="command", required=False)
        for name in subcommand_names:
            sp = sub_temp.add_parser(name)
            _add_infer_args(sp)
        print(parser.format_help())
        print("\nSUBCOMMANDS:\n")
        for name, sp in sub_temp.choices.items():
            print(f"== {name} ==")
            print(sp.format_help())
        return

    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run LEADER batch inference",
        description="Extract minutiae from fingerprint images using LEADER",
    )
    _add_infer_args(infer_parser)
    infer_parser.set_defaults(func=infer_command)

    forward_parser = subparsers.add_parser(
        "forward",
        help="Alias for infer",
        description="Extract minutiae from fingerprint images using LEADER (alias for infer)",
    )
    _add_infer_args(forward_parser)
    forward_parser.set_defaults(func=infer_command)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
