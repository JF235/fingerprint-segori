# pyfing.pytorch

PyTorch adaptation layer for `pyfing`, isolated from the original code.

## Purpose

This submodule adds:

- PyTorch architectures for `pyfing` neural models;
- Keras weights conversion (`*.weights.h5`) to PyTorch (`*.pth`);
- Keras <-> PyTorch compatibility validation:
  - tensor-by-tensor comparison of converted weights;
  - inference output comparison on deterministic fixtures.

Covered models:

- `SUFS`
- `SNFOE`
- `SNFFE`
- `SNFEN`
- `LEADER`

## Structure

- `common.py`: shared blocks and conversion utilities.
- `*_model.py`: PyTorch architectures per model.
- `algorithms.py`: wrappers with `run`/`run_on_db` signatures.
- `simple_api.py`: simple API mirroring the main API.
- `api.py`: high-throughput batch inference pipeline for LEADER (multi-GPU).
- `cli.py`: `leader` CLI entry point.
- `registry.py`: central model registry and weight paths.
- `compat.py`: conversion and validation core.
- `tools/convert_weights.py`: conversion CLI.
- `tools/validate_compat.py`: validation CLI.
- `models/`: default destination for `.pth` files and `manifest.json`.

## Weight conversion

From the repository root:

```bash
python -m pyfing.pytorch.tools.convert_weights --all
```

Convert only one model:

```bash
python -m pyfing.pytorch.tools.convert_weights --model snfoe
```

## Compatibility validation

Validate all models:

```bash
python -m pyfing.pytorch.tools.validate_compat --all
```

Save JSON report:

```bash
python -m pyfing.pytorch.tools.validate_compat --all --report-json /tmp/pyfing_pytorch_compat.json
```

## Using the PyTorch API

```python
import pyfing.pytorch as pft

seg = pft.fingerprint_segmentation(fingerprint)
ori = pft.orientation_field_estimation(fingerprint, seg)
rp = pft.frequency_estimation(fingerprint, ori, seg)
enh = pft.fingerprint_enhancement(fingerprint, ori, rp, seg)
mnt = pft.minutiae_extraction(fingerprint)
```

## LEADER batch inference CLI

Install the package in editable mode (one-time):

```bash
pip install -e .
```

### Basic usage

```bash
# Single GPU
leader infer images/ output/ --gpus 1 -b 8

# 4-GPU DDP
leader infer images/ output/ --gpus 4 -b 16 --recursive

# Specific physical GPUs
leader infer images/ output/ --gpus [2,3] -b 8

# CPU only
leader infer images/ output/ --gpus 0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gpus` | `1` | `0`=CPU, `N`=N-GPU DDP, `[i,j]`=specific GPUs |
| `-b / --batch-size` | `8` | Images per GPU per step |
| `--cores` | `4` | DataLoader worker processes per GPU |
| `--cpu-workers` | `4` | Threads for saving / post-processing |
| `--dpi` | `500` | Input DPI; images are pre-scaled to 500 dpi |
| `--threshold` | `0.6` | Minutia quality threshold (0–1) |
| `--type-threshold` | `0.5` | Ending vs Bifurcation decision threshold |
| `--strategy` | `full_gpu` | `full_gpu` or `hybrid` (GPU infer + CPU post) |
| `-r / --recursive` | off | Recurse into subdirectories |
| `--compile` | off | `torch.compile` (experimental) |
| `--max-dim` | `1024` | Max image dimension after DPI scaling |
| `--weights` | bundled | Path to custom `.pth` weights |

### Output format

```
output/
└── minutiae/
    ├── image1.min
    └── image2.min
```

Each `.min` file follows the standard format:

```
#MIN X Y ANGLE QUALITY TYPE TYPESCORE
123 456 270 82 1 91
...
```

- `X`, `Y`: coordinates in pixels (origin top-left)
- `ANGLE`: CCW degrees in `[0, 360)`, zero pointing right
- `QUALITY`: integer `0–100`
- `TYPE`: `1` = ridge ending, `2` = ridge bifurcation
- `TYPESCORE`: raw LEADER type confidence as integer `0–100`

### Angle convention

LEADER outputs directions as CCW radians (standard math convention).
Conversion to `.min`: `round(degrees(direction_rad) % 360)` — **no negation**
(unlike FingerNet which outputs CW radians and uses `round((-degrees(r)) % 360)`).

### Python API

```python
from pyfing.pytorch.api import run_inference

run_inference(
    input_path="images/",
    output_path="output/",
    gpus=2,
    batch_size=8,
    dpi=500,
    threshold=0.6,
    recursive=True,
)
```

## Notes

- This submodule is separate by design and does not change the original `pyfing` API.
- Conversion/validation scripts assume an environment with `torch`, `keras`, `tensorflow`, and `h5py` installed.
- The batch inference pipeline (`api.py` / `leader` CLI) requires only `torch` — no Keras/TF needed.
