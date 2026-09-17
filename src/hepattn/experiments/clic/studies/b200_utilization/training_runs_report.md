# CLIC Training Runs — HiPerGator Report

All original jobs submitted **2026-06-05**; the 4× B200 resubmit ² was submitted **2026-06-09**. Training config: 200 epochs, `bf16-mixed` precision, `flash-varlen` attention, Lion optimizer.

| | **4× B200** | **4× B200 (resubmit)** ² | **1× B200** | **3× L4 (1 node)** | **6× L4 (2 nodes)** |
|---|---|---|---|---|---|
| **GPU VRAM** | 4× 192 GB | 4× 192 GB | 1× 192 GB | 3× 24 GB | 6× 24 GB |
| **Nodes** | 1 | 1 | 1 | 1 | 2 |
| **CPUs/task** | 16 | 16 | 16 | 16 | 16 |
| **Requested CPU RAM:** | 200 GB ¹ | 300 GB | 60 GB | 150 GB | 150 GB/node |
| **Time limit** | 48 h | 48 h | 72 h | 168 h | 168 h |
| **Batch size/GPU** | 2048 | 2048 | 2048 | 256 | 256 |
| **Global batch size** | 8192 | 8192 | 2048 | 768 | 1536 |
| **Steps/epoch** | — | 122 | 486 | 1295 | 648 |
| **Speed (it/s)** | — | 0.38 | 0.40 | 1.81 | 1.74 |
| **Time/epoch** | — | ~5.3 min | ~20 min | ~12 min | ~6 min |
| **Throughput (samples/s)** | — | ~3115 | ~818 | ~1393 | ~2676 |
| **Queue time** | 32 h 42 min | 3 d 0 h 37 min ² | 12 h 59 min | 26 min | 2 h 48 min |
| **Run time** | 43 min | 18 h 07 min | 2 d 20 h 17 min | 1 d 15 h 46 min | 20 h 49 min |
| **Outcome** | OOM crash ¹ | ✅ 200 epochs | ✅ 200 epochs | ✅ 200 epochs | ✅ 200 epochs |

¹ The 4× B200 job OOM-crashed at epoch 0 step 50/122. Root cause: `--mem=200G` is shared across 4 DDP tasks (50 GB/task), less than the ~60 GB/task needed. Fixed to `--mem=300G` and resubmitted as job 34185716.

² The resubmit (job 34185716) was submitted **2026-06-09 09:49** — not 2026-06-05 like the others.

## Hardware Efficiency: why 3× L4 beat 1× B200

A surprising result: **three L4 GPUs trained faster than one B200** (3× L4 ≈ 12 min/epoch and
finished in ~40 h, vs 1× B200 ≈ 20 min/epoch over ~68 h). The B200 is a flagship data-center
GPU; the L4 is a small, cheap inference card. So why did the "weaker" hardware win?

### What the numbers show
The fair way to compare two GPUs is **throughput per GPU** (training samples processed per
second, per GPU), which strips out the queue and the different batch sizes:

| | samples/s (total) | ÷ GPUs = **per GPU** |
|---|---|---|
| 1× B200 | ~818 | **~818 / GPU** |
| 3× L4 | ~1393 | **~464 / GPU** |

Per GPU, one B200 is only **~1.76× faster than one L4**. Since three L4s add up to
`3 × 464 ≈ 1393`, they collectively out-run a single B200 (~818) by ~1.7× — which is exactly
the wall-clock gap we saw.

### Why this means the B200 was underutilized
"FLOPS" (floating-point operations per second) is a GPU's raw math horsepower — how many
multiply/add calculations it can do each second. On paper (spec-sheet "peak" FLOPS in the
`bf16` number format these runs use), a **B200 has roughly 15–40× the horsepower of an L4** —
i.e. one B200 ≈ 15–40 L4s' worth of raw compute.

But in these runs it delivered only **1.76×** the real-world speed of an L4. Paying for 15–40×
the horsepower and getting ~1.8× the speed means **the GPU's math was not the bottleneck** —
the B200 spent most of its time waiting, not computing. 

This underutilization is expected for **this** workload: the model is tiny (10.1 M parameters),
which doesn't come close to filling a B200's compute, and the training step does real CPU-side
data loading/preprocessing each iteration. Adding more GPUs also adds more parallel data
pipelines (each DDP task runs its own 16 workers), so more-but-weaker GPUs win here.

### What is proven vs. still a hypothesis
- **Proven** (from the measured throughput): the B200 is heavily underutilized on this
  workload — something other than its raw compute is the ceiling.
- **Not yet proven**: the *exact* bottleneck. The leading suspect is the data pipeline
  (CPU/disk feeding the GPU), but it could instead be launch overhead from many tiny
  operations on a small model, host-side Python overhead, or memory bandwidth. The runs here
  had the profiler disabled (`profiler: null`) and logged no GPU-utilization data, so the
  mechanism can't be confirmed from existing logs.
- **How to confirm**: re-run a short 1-GPU job with `profiler: simple` (Lightning then reports
  time spent in `train_dataloader_next`, i.e. waiting for data) plus a synthetic-data control
  (in-memory random tensors, no I/O) and a `num_workers` sweep. If throughput jumps with
  synthetic data / more workers, it's data-pipeline-bound.

### Practical takeaway
For this small model + I/O-heavy pipeline, the B200 is poor value — you pay for compute you
can't feed. Several cheap L4s beat it on both wall-clock time and cost. A B200 would only pay
off if the model were scaled up, the per-GPU batch increased, or the data pipeline made cheap
enough that the GPU itself becomes the limiting factor.

## Issues Encountered and Fixes

### 1. Truncated validation file
`val_clic_fix.root` was downloaded with only 233 MB out of an expected 296 MB. The ROOT header reported `fEND=309 MB` while the actual file was shorter, causing uproot to fail with `received 0 bytes from FSSpecSource`. Re-downloaded from CERNBox.

### 2. Comet ML requires API key
`MyCometLogger` was failing at startup with `Comet.ml requires an API key` because `COMET_API_KEY` was not set. Fixed by modifying `src/hepattn/utils/loggers.py` to default to `online=False` (offline mode) when no API key is present:
```python
if not os.environ.get("COMET_API_KEY") and kwargs.get("online") is None:
    kwargs["online"] = False
```

### 3. FlashAttention dtype error on L4 GPUs
`flash-varlen` attention was receiving fp32 tensors on L4 GPUs despite `bf16-mixed` precision, because `torch.compile` does not correctly propagate the AMP autocast context into the compiled graph. Fixed in `src/hepattn/models/attention.py` with an explicit cast:
```python
if q_flat.dtype == torch.float32:
    q_flat, k_flat, v_flat = q_flat.to(torch.bfloat16), ...
```

### 4. OOM crash on 4× B200
The 4-GPU job OOM-killed ranks 1–3 at epoch 0 step 50/122. Root cause: SLURM's `--mem` is per-node total CPU RAM. With 4 DDP ranks each spawning 16 DataLoader workers (64 workers total), the CPU RAM requirement is ~4× the single-GPU case (~240 GB), but only 200 GB was requested. Fixed by increasing `--mem` to 300 GB in `submit_training_hpg.sh`.
