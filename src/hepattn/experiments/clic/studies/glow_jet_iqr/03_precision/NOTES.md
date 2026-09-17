# Task 3 — fp32 (32-true) training + confirm paper precision

**Status:** 🟡 fp32 training SUBMITTED (job 36518920, PENDING as of 2026-07-07) · paper-precision check ⬜ TODO

## Setup log (2026-07-07)
- Config: **`configs/clic_v6_fp32.yaml`** — full copy of `base.yaml` (v6, 10.1M) with ONLY
  precision knobs changed, so precision is the isolated variable:
  - `trainer.precision: 32-true` (was `bf16-mixed`)
  - encoder `attn_type: torch` (was `flash-varlen` — no true-fp32 path; also sidesteps the
    L4 bf16-cast in `attention.py`)
  - `matmul_precision: highest` (was `high`/TF32 → genuine full-fp32 matmuls; matches eval.yaml)
  - `batch_size: 128`/GPU (was 256 in bf16; fp32 ~2× activation mem). Global = 128×6 = **768**,
    same global batch as the plotted 3×L4 run → directly comparable.
- Submitted on **6×L4 (2 nodes × 3)** via `submit_training_hpg_l4_2nodes.sh configs/clic_v6_fp32.yaml`
  → **job 36518920** (script forces `--trainer.devices=3 --trainer.num_nodes=2`).
- Output folder will be `logs/clic_v6_fp32_<timestamp>`.
- ⚠️ **Watch first ~50 steps for OOM** (batch 128 in fp32 is a first estimate); if it OOMs, drop to 96 or 64.

## Objective
Isolate **training precision** as a possible cause of the inverted IQR trend. All current
runs train at `bf16-mixed` (`../../configs/base.yaml`). Retrain at full **fp32 (`32-true`)**
and compare the IQR trend. Separately, confirm what precision the **paper** used.

## Two sub-tasks

### 3a. Confirm the paper's floating-point implementation
- Paper: GLOW, arXiv:2508.20092 (linked in `../../README.md`). Phase-1 notes recorded
  precision as **"not stated in paper"** — re-check the paper + its appendix/config for any
  mention (fp32 / bf16 / amp / mixed), and check the upstream `samvanstroud/hepattn` CLIC
  config at the paper's commit (overlaps with task 4). Record the finding here.

### 3b. Train at fp32
- Make a config override (e.g. `configs/clic_v?_fp32.yaml` or `--trainer.precision=32-true`)
  from `base.yaml` (v6, 10.1M — match the plotted model's size so precision is the ONLY
  变量).
- **Watch out — flash-varlen attention:** `base.yaml` uses flash-varlen, and the local
  `attention.py` L4 fix casts flash inputs to **bf16 when they arrive as fp32** (guarded by
  `if q_flat.dtype == torch.float32`). Under `32-true` this cast would *defeat* the point.
  Two options: (i) set `attn_type: torch` (standard attention, true fp32, slower) — cleanest
  for an apples-to-apples precision test; or (ii) verify flash-varlen actually runs fp32.
  **Prefer `attn_type: torch` + `32-true`** so the run is genuinely full-precision end to end.
- fp32 ~2× the memory/time of bf16 → expect smaller batch / more GPUs / longer wall-clock.
- 200 epochs, same data/optimizer/schedule as v6. New folder under `logs/`.

## Method (compare)
Eval the fp32 run (`../../submit_eval_run.sh`) and overlay its IQR vs the bf16 v6 runs with
`../00_cross_run/compare_runs_iqr.py`. If the trend is unchanged → precision is not the cause.

## Checklist
- [ ] paper precision confirmed (or confirmed unstated) + upstream config checked
- [ ] fp32 config written (32-true, attn_type torch)
- [ ] training submitted → 200 epochs
- [ ] eval'd → IQR overlay vs bf16
- [ ] verdict (precision matters? Y/N)

## Results
_(fill in)_
