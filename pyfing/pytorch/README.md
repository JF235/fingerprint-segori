# pyfing.pytorch

Camada de adaptação PyTorch para o `pyfing`, isolada do código original.

## Objetivo

Este submódulo adiciona:

- arquiteturas PyTorch para os modelos neurais do `pyfing`;
- conversão de pesos Keras (`*.weights.h5`) para PyTorch (`*.pth`);
- validação de compatibilidade Keras <-> PyTorch:
  - comparação tensor-a-tensor dos pesos convertidos;
  - comparação de saída de inferência em fixtures determinísticas.

Modelos cobertos:

- `SUFS`
- `SNFOE`
- `SNFFE`
- `SNFEN`
- `LEADER`

## Estrutura

- `common.py`: blocos compartilhados e utilitários de conversão.
- `*_model.py`: arquiteturas PyTorch por modelo.
- `algorithms.py`: wrappers com assinaturas de `run`/`run_on_db`.
- `simple_api.py`: API simples espelhando a API principal.
- `registry.py`: registro central de modelos e caminhos de pesos.
- `compat.py`: núcleo de conversão e validação.
- `tools/convert_weights.py`: CLI de conversão.
- `tools/validate_compat.py`: CLI de validação.
- `models/`: destino padrão dos arquivos `.pth` e `manifest.json`.

## Conversão de pesos

Na raiz do repositório:

```bash
python -m pyfing.pytorch.tools.convert_weights --all
```

Converter apenas um modelo:

```bash
python -m pyfing.pytorch.tools.convert_weights --model snfoe
```

## Validação de compatibilidade

Validar todos os modelos:

```bash
python -m pyfing.pytorch.tools.validate_compat --all
```

Salvar relatório JSON:

```bash
python -m pyfing.pytorch.tools.validate_compat --all --report-json /tmp/pyfing_pytorch_compat.json
```

## Uso da API PyTorch

```python
import pyfing.pytorch as pft

seg = pft.fingerprint_segmentation(fingerprint)
ori = pft.orientation_field_estimation(fingerprint, seg)
rp = pft.frequency_estimation(fingerprint, ori, seg)
enh = pft.fingerprint_enhancement(fingerprint, ori, rp, seg)
mnt = pft.minutiae_extraction(fingerprint)
```

## Observações

- Este submódulo é separado por design e não altera a API original do `pyfing`.
- Os scripts de conversão/validação assumem ambiente com `torch`, `keras`, `tensorflow` e `h5py` instalados.
