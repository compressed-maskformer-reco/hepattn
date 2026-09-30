# CLIC Training Log

Index of **all** CLIC training runs. One row per run. Add a new row (and a per-run
note) each time a training is submitted.

- **Config** names the config used — see [`configs/CONFIGS.md`](./configs/CONFIGS.md)
  for what each config contains.
- Output folders live under `logs/` and are named `<config-name>_<YYYYMMDD>-T<HHMMSS>`,
  where the timestamp is the job **start** time (not submit time).
- For folders lacking notes, run identity can be recovered from each folder's
  `metadata.yaml` (`slurm_job_id`, `num_gpus`, `batch_size`, `hostname`) and
  `config.yaml` (`num_nodes`, `devices`), cross-checked against checkpoint mtimes.

## Runs

| Folder | Config | SLURM job | Hardware | Nodes × GPU/node | Batch/GPU | Global batch | Outcome |
|---|---|---|---|---|---|---|---|
| `clic_v6_20260606-T194653` | v6 | 33954038 | B200 192 GB | 1 × 4 | 2048 | 8192 | ❌ OOM crash at epoch 0, step 50 |
| `clic_v6_20260606-T000306` | v6 | 33954039 | B200 192 GB | 1 × 1 | 2048 | 2048 | ✅ 200 epochs |
| `clic_v6_20260605-T113014` | v6 | 33954040 | L4 24 GB | 1 × 3 | 256 | 768 | ✅ 200 epochs |
| `clic_v6_20260605-T143447` | v6 | 33955412 | L4 24 GB | 2 × 3 | 256 | 1536 | ✅ 200 epochs |
| `clic_v6_20260612-T102707` | v6 | 34185716 | B200 192 GB | 1 × 4 | 2048 | 8192 | ✅ 200 epochs |

## Per-run notes

### `clic_v6_20260606-T194653` — 4× B200 (OOM)
- Job `33954038`, host `c1109a-s15.ufhpc`, 1 node × 4 GPU, batch 2048/GPU.
- **OOM-crashed at epoch 0 step 50/122.** Root cause: `--mem=200G` shared across 4 DDP
  tasks (50 GB/task) vs the ~60 GB/task needed. No checkpoints written.
- Left a 35 GB core dump in the clic experiment root (`core.pt_nccl_watchdg-...`). Safe
  to delete once no longer needed.
- Fixed (`--mem=300G`) and resubmitted as job 34185716 → `clic_v6_20260612-T102707`.

### `clic_v6_20260606-T000306` — 1× B200
- Job `33954039`, host `c0906a-s15.ufhpc`, 1 node × 1 GPU, batch 2048.
- ✅ Completed 200 epochs. ~0.40 it/s, ~20 min/epoch, run time 2 d 20 h 17 min.

### `clic_v6_20260605-T113014` — 3× L4, 1 node
- Job `33954040`, host `c0609a-s13.ufhpc`, 1 node × 3 GPU, batch 256, global 768.
- ✅ Completed 200 epochs (ckpts epoch 0–199). ~1.81 it/s, ~12 min/epoch.
- Checkpoints span 2026-06-05 11:46 → 2026-06-07 03:15 (≈ 1 d 15 h 30 m).
- `ckpts/` contains **202 files = 200 checkpoints (epoch 0–199) + 2 evaluation
  outputs**. The extras share the `epoch=195` prefix because eval was run from the best
  checkpoint (epoch 195, val_loss 3.78800 — the global minimum):
  `epoch=195-val_loss=3.78800__test.h5` (2.8 GB) and `..__test.root` (67 MB), both
  written 2026-06-09. This is the only run with eval outputs stored alongside its ckpts.

### `clic_v6_20260605-T143447` — 6× L4, 2 nodes
- Job `33955412`, host `c0611a-s4.ufhpc`, 2 nodes × 3 GPU, batch 256, global 1536.
- ✅ Completed 200 epochs (ckpts epoch 0–199). ~1.74 it/s, ~6 min/epoch.
- Checkpoints span 2026-06-05 14:45 → 2026-06-06 11:23 (≈ 20 h 38 m).

### `clic_v6_20260612-T102707` — 4× B200 (resubmit)
- Job `34185716`, host `c1103a-s15.ufhpc`, 1 node × 4 GPU, batch 2048, global 8192.
- Resubmit of the OOM'd `T194653` with `--mem=300G`. Submitted 2026-06-09 09:49, queued
  3 d 0 h 37 min, started 2026-06-12 10:26.
- ✅ Completed 200 epochs (ckpts epoch 0–199) in 18 h 07 min (`sacct` Elapsed).
  Checkpoints span 2026-06-12 10:37 → 2026-06-13 04:33.

> The five v6 runs above were a hardware/parallelism scaling study (1×/4× B200,
> 3×/6× L4 at fixed config). A detailed write-up lives in
> [`studies/b200_utilization/training_runs_report.md`](./studies/b200_utilization/training_runs_report.md),
> and the follow-on profiling plan (to confirm the B200 bottleneck) in
> [`studies/b200_utilization/profiling/`](./studies/b200_utilization/profiling/README.md).

## Conventions / housekeeping
- Folder timestamps are job **start** times; queue time sits between submit and start.
- The 35 GB core dump `core.pt_nccl_watchdg-...-c1109a-s15.ufhpc.3319738` in the
  experiment root is debris from the `T194653` OOM crash and can be removed.
