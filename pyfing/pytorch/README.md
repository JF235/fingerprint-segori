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

## Notes

- This submodule is separate by design and does not change the original `pyfing` API.
- Conversion/validation scripts assume an environment with `torch`, `keras`, `tensorflow`, and `h5py` installed.
