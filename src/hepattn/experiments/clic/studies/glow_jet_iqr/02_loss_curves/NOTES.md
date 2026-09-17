# Task 2 — Loss curves of the plotted 3×L4 run

**Status:** ⬜ TODO

## Objective
Plot train + validation loss vs epoch for the **plotted** model
(`logs/clic_v6_20260605-T113014`, 3×L4, the run behind the disputed IQR figure) to check
convergence — plateau, overfitting (train↓ while val↑), or LR-schedule pathology that a
single "final val_loss" number would hide.

## Why train-loss is missing but val-loss survives (definitive, checked 2026-07-07)
Two independent mechanisms:
- **val/loss survives** because the `Checkpoint` callback (`monitor: val/loss`) bakes the
  monitored value into every checkpoint **filename** (`epoch=N-val_loss=X.ckpt`) and into the
  ckpt file — filesystem-persisted, logger-independent. Recoverable for all runs.
- **train-loss goes ONLY to the logger** (`MyCometLogger`). On HPG compute nodes there is no
  internet and no `COMET_API_KEY` (submit scripts set `COMET_EXPERIMENT_KEY` but never `export`
  it and never set an API key), so `MyCometLogger` auto-falls back to **offline mode**
  (`loggers.py`: `online=False` when no key). The offline experiment IS created (the progress
  bar shows `v_num=<key>`), but **no offline archive lands on disk** — the run folders contain
  only ckpts/config/metadata/times, no `.zip`, no `metrics.csv`. Likely cause: offline comet's
  working dir goes under `TMPDIR=/var/tmp/` (set in the submit scripts), which is container-
  ephemeral and lost at job end. Net: **Comet is effectively off on HPG, and train-loss is not
  persisted anywhere.** So val-loss = free from ckpt names; train-loss = currently unrecoverable
  for past runs (v6 + the running v7 jobs).

## ✅ FIX LANDED 2026-07-07 — CSVLogger added in cli.py
Implemented in `src/hepattn/utils/cli.py` `after_instantiate_classes` (fit branch): appends a
`lightning.pytorch.loggers.CSVLogger(save_dir=default_root_dir, name="csv_metrics", version="")`
to `self.trainer.loggers`. Done POST-instantiation so it doesn't disturb the single-logger
`name`/`offline_directory` parser wiring (link_arguments line 62 / injection line 86) — no other
experiment breaks. Validated standalone: writes `<run_dir>/csv_metrics/metrics.csv` with columns
`epoch,step,train/loss,val/loss,...` (wrapper.py logs `self.log(f"{stage}/loss")` for train+val).
Train loss is per-step, val per-epoch — both carry an `epoch` column. Since fork jobs run live
code from /blue (container binds it), the **already-queued fp32 job 36518920 will pick this up**
when it starts. NOTE: the paper-tag clone has its own cli.py + plain CometLogger — apply the same
fix there separately if train-loss is wanted for that run (val-loss from ckpt names covers it).

## Fix for FUTURE runs (answering "can this be fixed?")
The clean fix is a **`CSVLogger`** (writes `metrics.csv` with train+val loss to disk, no
internet). BUT the repo's custom `utils/cli.py` hardcodes a **single** logger — it does
`link_arguments("name", "trainer.logger.init_args.name")` (line 62) and force-injects
`offline_directory` into `trainer.logger` (line 86). So `trainer.logger` cannot be made a list
or swapped to `CSVLogger` (no `offline_directory` arg) without a small `cli.py` change that:
(a) guards the `name` link / offline_directory injection to loggers that accept them, or
(b) iterates when `logger` is a list. This is a shared-infra change → make + test locally
(jsonargparse parse) before relying on it. Until then, val-loss from ckpt names remains the
fallback for every run. **The fp32 run (task 3) currently keeps the single MyCometLogger** to
avoid risking the run — its train-loss will also be missing unless we land the cli.py fix first.

## What is available (checked 2026-07-07)
- The run dir has **NO local metrics files** (no CSV / tfevents / comet-offline) — only
  `ckpts/`, `config.yaml`, `metadata.yaml`, `times/`.
- Logger is `hepattn.utils.loggers.MyCometLogger` → metrics went to **Comet online**,
  project **`hepattn-clic`**.
- **Validation loss IS recoverable locally for free:** every checkpoint is named
  `epoch=<N>-val_loss=<X>.ckpt` for epoch 0–199. Parse the 200 filenames → full val-loss
  curve, no Comet needed. (Sanity: epoch=100 → 4.06470, epoch=199 → 3.79233, best epoch=195 → 3.788.)

## Method
1. **Val-loss curve (do first, zero dependencies):** glob `ckpts/epoch=*-val_loss=*.ckpt`,
   regex out (epoch, val_loss), sort, plot. Mark the best (epoch 195). Output PNG here.
2. **Train-loss curve (needs Comet):** pull from Comet project `hepattn-clic` — via the
   Comet web UI export, or `comet-ml` Python API with an API key + the experiment key for
   this run. Overlay train vs val on one axis. If Comet is inaccessible, val-only is a
   usable deliverable; note the gap.
3. Optionally do the same for the other v6 runs for comparison (all have named ckpts).

## Checklist
- [ ] val-loss-vs-epoch plotted from ckpt names
- [ ] Comet access sorted → train-loss pulled (or documented as unavailable)
- [ ] train vs val overlay
- [ ] convergence verdict (converged? overfit? plateau?)

## Results
_(fill in)_
