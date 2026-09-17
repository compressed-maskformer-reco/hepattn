# Task 1 — 700k small model (v7) training + IQR eval

**Status:** 🟡 2-node run FINISHED; eval running (job 36520192) → plotting. 1-node still running.

## Job→folder mapping (VERIFIED via each folder's `metadata.yaml` slurm_job_id) — 2026-07-07
The two folders both start `clic_v7_20260706-T18141…` (both jobs started 18:14); the timestamp
does NOT tell them apart. The definitive mapping:
| Folder | slurm_job_id | Job | Config | State | ckpts | Best ckpt |
|---|---|---|---|---|---|---|
| `clic_v7_20260706-T181417` | 36450143 | **2-node** | v7 | **FINISHED (200 ep)** | 200 (0–199) | **epoch=192, val_loss 4.33644** |
| `clic_v7_20260706-T181418` | 36472892 | **1-node** | v7 | RUNNING (~ep 109) | 109 (0–108) | tbd when done |

> ⚠️ Earlier I mis-assigned these by folder timestamp and evaluated T181418 (the *running*
> 1-node run, partial epoch 105) — cancelled (job 36519705). There is **NO checkpoint anomaly**:
> T181418 only has epochs 0–108 because it is still at ~epoch 109. The finished 2-node run
> (T181417) has all 200 checkpoints normally.

## Plan (triple-check: 2-node v7 + 1-node v7 + v6 family, per user)
- **2-node (36450143):** eval best ckpt **epoch=192 (val_loss 4.33644)** → **job 36520192** (1×L4)
  → `logs/clic_v7_20260706-T181417/ckpts/epoch=192-val_loss=4.33644__test.root`.
- **1-node (36472892):** eval its best ckpt once it finishes 200 epochs.
- Plot: **`plot_v7_iqr.py`** (auto-discovers each run's `*__test.root`, SKIPs un-evaled ones) →
  `jet_iqr_v7_vs_v6.png`. Re-run as each eval lands:
  `pixi run -e clic python studies/glow_jet_iqr/01_small_model_v7/plot_v7_iqr.py`.
- (v7 best val_loss ~4.34 vs v6 ~3.79 — expected for a 14× smaller model.)

## Objective
Test whether the inverted high-E jet-IQR trend depends on model **capacity**. v7 is the
same architecture as v6 but **~14× smaller** (702,395 vs 10,126,115 params; `dim 256→64`,
heads `16→8`, narrower head MLPs — see `../../configs/CONFIGS.md`). Everything else
(6-enc/4-dec depth, 150 queries, 8 registers, Lion @ 8e-5, 200 epochs, bf16-mixed,
flash-varlen, same data) is identical.

**Interpretation key:** if v7's IQR trend is *the same* inverted shape → capacity is NOT
the driver, and the 12M-vs-10.1M lead weakens. If v7 is *dramatically worse at high E* →
capacity matters and the paper's larger 12M model is a plausible explanation.

## Runs (both v7, started 2026-07-06 T18:14)
| Job | Nodes×GPU | Global batch | Folder |
|---|---|---|---|
| 36472892 | 1×3 L4 | 512×3 = 1536 | `logs/clic_v7_20260706-T181417` |
| 36450143 | 2×3 L4 | 512×6 = 3072 | `logs/clic_v7_20260706-T181418` |
(submit scripts: `../../submit_training_hpg_l4.sh`, `../../submit_training_hpg_l4_2nodes.sh`)

## Method (when training finishes)
1. Confirm 200 epochs completed; pick best ckpt by `val_loss` (in filename).
2. Eval: `../../submit_eval_run.sh <run_folder> <best_ckpt>` → `__test.{h5,root}`
   (uses `eval.yaml`: `precision 32-true`, `attn_type torch`, `matmul highest`).
3. Overlay IQR vs the v6 runs + Pandora with `../00_cross_run/compare_runs_iqr.py`
   (add the v7 root paths; repoint `OUT` here). Bin `e_rel` in truth jet E, IQR = p75−p25.
4. Record the high-E IQR (~170–190 GeV bin) for both v7 runs in the table below.

## Checklist
- [ ] 1-node v7 (36472892) reached 200 epochs
- [ ] 2-node v7 (36450143) reached 200 epochs
- [ ] eval'd both → `.root` outputs
- [ ] IQR overlay produced
- [ ] verdict recorded (capacity matters? Y/N)

## Results — 2-node v7 (epoch 192) DONE 2026-07-07 → capacity is NOT the cause
Figure: `jet_iqr_v7_vs_v6.png` (v7 2-node vs v6 family vs Pandora; 1-node pending).

Jet-E response IQR (p75−p25) vs truth jet E [GeV]:
| E | 10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 | 170 | 190 |
|---|---|---|---|---|---|---|---|---|---|---|
| v7 700k (2-node) | 0.082 | 0.072 | 0.076 | 0.086 | 0.087 | 0.097 | 0.090 | 0.095 | 0.099 | 0.101 |
| 3×L4 v6 (plotted) | 0.075 | 0.067 | 0.070 | 0.077 | 0.077 | 0.085 | 0.077 | 0.083 | 0.099 | 0.096 |
| Pandora | 0.091 | 0.076 | 0.068 | 0.064 | 0.061 | 0.063 | 0.059 | 0.060 | 0.056 | 0.054 |

**The 700k v7 shows the SAME inverted rising trend** (~0.08→0.10) as v6, just a ~0.005–0.015
constant offset worse (expected for 14× fewer params) — NOT a change in shape. Shrinking the
model 10.1M→0.7M does not flip or flatten the trend. → **Capacity is not the driver**; the
paper's 12M-vs-our-10.1M gap is a red herring. Consistent with task 4 (cause = post-paper model
refactor). Task 5 (paper-tag retrain) is the confirming test.

TODO: eval the 1-node v7 (36472892) when it finishes 200 epochs → re-run `plot_v7_iqr.py` to add
it (auto-discovered; the partial epoch-105 root was renamed to `*.PARTIAL-1node-stilltraining`).

| Run | best val_loss | IQR @ ~10 GeV | IQR @ ~90 | IQR @ ~170 | IQR @ ~190 |
|---|---|---|---|---|---|
| v7 1-node | | | | | |
| v7 2-node | | | | | |
| v6 3×L4 (ref) | 3.788 | 0.075 | 0.077 | 0.099 | 0.096 |
| Pandora (ref) | — | 0.091 | 0.061 | 0.056 | 0.054 |
