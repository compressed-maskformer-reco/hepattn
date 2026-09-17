# CLIC Configs — Index & Comparison

This directory holds the training/eval configs for the CLIC experiment. Each training
config sets `name: clic_v<N>`, which becomes the prefix of its output folder under
`logs/` (`clic_v<N>_<YYYYMMDD>-T<HHMMSS>`). See
[`../TRAINING_LOG.md`](../TRAINING_LOG.md) for which runs used which config.

## Files
| File | Purpose |
|---|---|
| `base.yaml` | **v6** — production baseline (`name: clic_v6`). |
| `clic_v7.yaml` | **v7** — small model (`name: clic_v7`). |
| `clic_v6_fp32.yaml` | **v6 fp32** — full-precision variant of `base.yaml` for the precision study (`name: clic_v6_fp32`): `precision 32-true`, encoder `attn_type torch`, `matmul_precision highest`, `batch_size 128`. Everything else identical to v6. See `../studies/glow_jet_iqr/03_precision/`. |
| `clic_var_transform.yaml` | Input/target scaling dictionary (referenced by both via `scale_dict_path`). |
| `eval.yaml`, `test_override.yaml` | Evaluation / test-time overrides, not training. |

## v6 vs v7

Both share the same architecture (MaskFormer: Fourier posenc → 6-layer encoder →
4-layer mask decoder → 4 task heads), 150 object queries, 8 register tokens, Lion
optimizer (max LR `8e-5`), 200 epochs, `bf16-mixed`, `flash-varlen` attention. They
differ in **model width** (and the batch/device settings tuned to it):

| | **v6** (`base.yaml`) | **v7** (`clic_v7.yaml`) |
|---|---|---|
| Trainable params | **10,126,115** | **702,395** |
| Embedding dim | 256 | 64 |
| Attention heads (enc & dec) | 16 | 8 |
| Class head hidden | `[256, 128, 32]` | `[64, 128, 32]` |
| Regression head input / hidden | 518 / `[512, 256, 128, 64, 32]` | 134 / `[128, 128, 128, 64, 32]` |
| `batch_size` | 2048 | 512 |
| `trainer.devices` | 4 | 2 |
| `num_val` | 5000 | 5000 |
| Data root | `/blue/avery/m.mazza/.../data/clic/` | `/blue/avery/m.mazza/.../data/clic/` |

Encoder/decoder depth, query count, register tokens, optimizer, schedule, epochs, and
data are identical. v7 is ~14× smaller purely from the `dim: 256 → 64` reduction (heads
and head-MLP widths scale with it).
