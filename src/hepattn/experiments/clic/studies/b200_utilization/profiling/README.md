# CLIC Training Profiling Study

**Status:** Planned, not started (deferred — low priority while cluster resources are in use).

**Goal:** Find out *why the B200 GPU is underutilized* during CLIC training, so we
know whether the bottleneck is the data pipeline, host-side Python/CPU work, the
CPU Hungarian matcher, kernel launch overhead, or something else — and then fix it.

This file is a self-contained handoff for a **fresh Claude session** (or a human).
It has all the context, exact commands, file paths, and experimental-design rules
needed to execute the study without re-doing the investigation.

---

## 1. Background & motivation

See [`../training_runs_report.md`](../training_runs_report.md) for the full
comparison. Key facts:

- Across 5 HiPerGator runs (4×/1× B200, 3×/6× L4), **1 B200 was only ~1.76× faster
  than 1 L4 per GPU**, despite the B200 having ~15–40× the peak bf16 FLOPS. Three
  cheap L4s beat a single B200 on wall-clock.
- **Proven:** the B200 is heavily underutilized on this workload — raw compute is
  *not* the ceiling. The model is tiny (10.1 M params) and each step does real
  CPU-side data loading/preprocessing.
- **Not yet proven:** the *exact* bottleneck. Leading suspect is the data pipeline
  (CPU/disk feeding the GPU), but it could be host-side Python overhead, the CPU
  scipy Hungarian matcher, or launch overhead from many tiny ops on a small model.
- The existing runs had `profiler: null` and logged no GPU-util data, so the
  mechanism cannot be confirmed from existing logs. **This study confirms it.**

---

## 2. What the codebase looks like (so you don't have to re-investigate)

- Training runs through **PyTorch Lightning `LightningCLI`**. The `Trainer` is built
  entirely from YAML, so **`trainer.profiler` is a config change, NOT a code change.**
  Confirmed: saved run configs show `profiler: null` — the slot is live and unused.
- Entrypoint: `src/hepattn/experiments/clic/main.py` → `CLI(model_class=MPflow, ...)`,
  invoked as `python main.py fit --config <cfg>`.
- CLI subclass `src/hepattn/utils/cli.py` does **not** touch the profiler — it passes
  straight through from YAML.
- Configs: `src/hepattn/experiments/clic/configs/`. `base.yaml` is always loaded;
  run configs (`clic_v7.yaml`, etc.) layer on top via `--config`. The `trainer:` block
  is at `base.yaml` lines ~39–74 and maps 1:1 to `lightning.pytorch.Trainer` kwargs.
- A ready-made **commented profiler template** exists in a sibling experiment:
  `src/hepattn/experiments/trackml/configs/tracking.yaml` lines ~86–92.
- DataLoader: `src/hepattn/experiments/clic/pflow_data.py`, `get_dataloader` at
  lines ~812–822. Currently: `num_workers=16`, `batch_size=2048` (base) / `512`
  (clic_v7), `pin_memory=True`. **Missing** (using PyTorch defaults):
  `persistent_workers` (→ False: workers respawn every epoch) and `prefetch_factor`
  (→ 2). These are the two cheap tuning knobs.
- Submit scripts (SLURM + Apptainer + pixi) in the clic dir:
  `submit_training_hpg_1gpu.sh` (single B200 — **use this as the base for profiling**),
  `submit_training_hpg.sh` (4× B200), `submit_training_hpg_l4*.sh`.

### Known gotchas (important)
- **`torch.compile` pollutes op-level traces.** The `Compile` callback (in `base.yaml`
  callbacks list) `torch.compile`s the encoder+decoder on train start. For Phase 2
  (PyTorchProfiler) either remove `hepattn.callbacks.Compile` from the callbacks list
  in the profiling config, or profile only *after* warmup steps.
- **CPU scipy Hungarian matcher** runs every step inside `model.loss`
  (`matcher.default_solver: scipy` in `base.yaml`). On a fast B200 this is a prime
  GPU-idle stall — watch for it specifically in Phase 2/3 traces.
- **`MyThroughputMonitor` callback (`callbacks/throughput_monitor.py`) is stale** — its
  `dummy_forward` uses an old input schema (`batch["hit"]`, `phi/theta/r`) that no longer
  matches the current `(inputs, targets)` batch format. Do NOT enable it as-is; it will
  crash. Ignore it or fix its dummy input first.
- **`tensorboard` is NOT installed** (logger is Comet, offline). For PyTorchProfiler use
  `export_to_chrome: true` and view traces at https://ui.perfetto.dev. Only add
  tensorboard/torch-tb-profiler to `pyproject.toml` if you want the TB profiler plugin.
- **`nsys` is not a Python dep** — it must exist in the Apptainer container. Verify with
  `apptainer run --nv .../pixi.sif nsys --version` before planning any nsys run.

---

## 3. Does profiling slow the job down?

- **SimpleProfiler (Phase 1):** negligible (~0–2%). Just wall-clock timers around
  existing hooks. Safe to leave on.
- **PyTorchProfiler / torch.profiler (Phase 2):** moderate–high *per profiled step*, so
  scope it to a **short window** (a schedule of a few active steps), never the full run.
- **nsys (Phase 3):** moderate, also run over a short window.

You never profile a full 200-epoch run. Total wall-clock cost of the whole study is
minutes of compute (the queue wait dominates).

---

## 4. Experimental-design rules (READ BEFORE RUNNING)

1. **Profile on 1 GPU first** to remove DDP/NCCL noise. `submit_training_hpg_1gpu.sh`
   already sets `--trainer.devices=1`.
2. **Keep the baseline sacred.** The runs in `training_runs_report.md` are the reference.
   Put all profiling/tuning changes in a **separate, throwaway** `configs/profile.yaml`
   layered via `--config` — do NOT edit `base.yaml` / `clic_v7.yaml`.
3. **Change one thing at a time.** Same GPU, batch size, step count, config — vary only
   the single knob under test, so any speedup is attributable.
4. **Measurement (Phase 1) vs. optimization (Phase 1b) are different.** Phase 1 only
   observes. Phase 1b changes dataloader behavior. Run Phase 1 → decide → maybe Phase 1b
   → re-run Phase 1 to measure the delta.
5. If a knob helps, **promote it deliberately** into the real config and note in
   `training_runs_report.md` "from run X onward, setting Y changed." If it doesn't help,
   throw `profile.yaml` away — baseline untouched.
6. Keep runs short: always add `--trainer.limit_val_batches=0` and cap steps/epochs.

---

## 5. The plan (execute in order; stop early if Phase 1 answers the question)

### Phase 0 — isolate
Clone `submit_training_hpg_1gpu.sh` → `studies/profiling/submit_profile_1gpu.sh`.
Base it on 1× B200. Only `--time`, `--mem`, and the trainer flags change vs. the
original (resources otherwise identical: 1 node, 1 B200, `--cpus-per-task=16`).

### Phase 1 — SimpleProfiler (DO THIS FIRST; near-zero overhead, config-only, no code)
**Answers: is it data-bound or compute-bound?**

1. Create `configs/profile.yaml`:
   ```yaml
   # layer on top of base.yaml (or clic_v7.yaml) via a second --config
   trainer:
     profiler: simple
     max_steps: 200
     limit_val_batches: 0
   ```
2. Run: `python main.py fit --config configs/base.yaml --config configs/profile.yaml --trainer.devices=1`
   (or append the flags directly on the CLI instead of a config file).
3. **Resources:** 1× B200, 16 CPU, 1 node, `--mem=60G`, **`--time=01:00:00`**.
   Expected real run time ~20 min (~8 min for 200 steps at ~0.40 it/s + ~10 min startup:
   container + loading the full ROOT file into RAM + compile warmup).
4. **Read the printed profiler table.** Look at:
   - `[_TrainingEpochLoop].train_dataloader_next` → **time waiting for data**.
   - `run_training_batch`, `training_step`, `backward`, `optimizer_step` → **compute**.
   - If `train_dataloader_next` is a large fraction → **data-bound** → go to Phase 1b.
   - If it's small and `training_step`/`backward` dominate → **compute-bound** → skip
     1b, go to Phase 2.

### Phase 1b — cheap data-pipeline fixes (ONLY if Phase 1 shows data-bound)
**Tests whether tuning the input pipeline recovers throughput.**

1. In `src/hepattn/experiments/clic/pflow_data.py` `get_dataloader` (~lines 812–822),
   add `persistent_workers=True` and expose/set `prefetch_factor` (try 4, then 8).
   Keep the change minimal and easily revertable.
2. **⚠️ Must cross an epoch boundary to see the `persistent_workers` benefit** (it saves
   the per-epoch worker respawn). 200 steps < 1 epoch (486 steps/epoch on 1× B200), so it
   would show nothing. Use `--trainer.max_epochs=2 --trainer.limit_val_batches=0`
   (≈ 970 steps) instead of `max_steps=200`.
3. **Resources:** 1× B200, 16 CPU, 1 node, **`--time=02:00:00`** (~50 min real).
   Memory: `--mem=60G` if `prefetch_factor=4`; **`--mem=100G` if `prefetch_factor=8`**
   (buffer grows to `16 workers × 8 = 128` batches of 2048 events padded to 160 nodes —
   a lot of pinned host RAM).
4. Re-run Phase 1 (SimpleProfiler) with these settings and compare `train_dataloader_next`
   against the Phase 1 baseline. Also try a `num_workers` sweep and a synthetic-data control
   (in-memory random tensors, no I/O) if you want to bound the maximum achievable throughput.

### Phase 2 — PyTorchProfiler (op-level CPU+CUDA, Chrome trace) — only if deeper detail needed
**Answers: within the GPU, is time real kernels vs. sync/copy vs. idle gaps?**

1. In the profiling config's `trainer:` block, replace `profiler: simple` with (mirror
   `trackml/configs/tracking.yaml:86`):
   ```yaml
   trainer:
     profiler:
       class_path: lightning.pytorch.profilers.PyTorchProfiler
       init_args:
         dirpath: ./studies/profiling/profile_logs/
         filename: clic_profile
         export_to_chrome: true
         record_shapes: true
         with_stack: false
         sort_by_key: cuda_time_total          # NOT cpu_time_total for a GPU-util question
         schedule: {wait: 1, warmup: 3, active: 5, repeat: 1}
   ```
2. **Remove `hepattn.callbacks.Compile` from the callbacks list** in the profiling config
   (torch.compile pollutes the trace), or do one pass with it off and one with it on.
3. View the Chrome/Perfetto JSON at https://ui.perfetto.dev. Look for
   `cudaStreamSynchronize` / `Memcpy HtoD` (sync/data), flash-attn kernels, and gaps
   between kernels (host-bound — e.g. the scipy matcher).

### Phase 3 — raw torch.profiler callback and/or nsys (deepest; only if Phase 2 insufficient)
- **Custom torch.profiler callback:** new `src/hepattn/callbacks/torch_profiler.py` modeled
  on the existing `inference_timer.py`, driving `torch.profiler.profile(...)` across
  `on_train_batch_start/end` with a schedule; register in `callbacks/__init__.py` and add
  to the config callbacks list. This is the only phase needing new code.
- **nsys (system-level, NCCL + full timeline):** edit a copy of the submit script to wrap
  the launch, e.g. `nsys profile -o /blue/.../clic_%j -t cuda,nvtx,osrt,cudnn,cublas ...`
  around the `apptainer run` command. **First verify `nsys` exists in the container**
  (see gotcha above). Keep steps ≤ ~50, profile 1 rank.

---

## 6. Deliverable

When done, append a **"Profiling results"** section to
[`../training_runs_report.md`](../training_runs_report.md) stating:
- the confirmed bottleneck (data / compute / matcher / launch overhead),
- the numbers that prove it (e.g. `train_dataloader_next` = X% of step time),
- any fix applied (which knob, the before/after throughput), and
- whether the fix was promoted into a real config (and from which run onward).

Also drop the raw profiler tables / trace files under `studies/profiling/profile_logs/`.
