# Running CLIC on HiPerGator (HPG)

HPG-specific instructions for training and evaluating the CLIC particle-flow model on
UF's HiPerGator cluster. For the model overview, evaluation flags, and data-format
details see [`README.md`](./README.md).

## Prerequisites

- **GPU partitions:**
  - `hpg-b200` — B200 192 GB (up to 4/node)
  - `hpg-turin` — L4 24 GB (up to 3/node)
- **Container + env:** the submit scripts run inside the pixi Apptainer image at the
  repo root (`pixi.sif`) via `apptainer run --nv ... pixi run -e clic ...`, so the `clic`
  pixi environment must be installed in the checkout you submit from. You do not need to
  enter the container by hand to submit.
- **CUDA module:** `module load cuda/12.8.1` (matches the container build; already in
  the submit scripts).
- **Data:** the CLIC ROOT files must be readable from the compute nodes.
  [`configs/hpg.yaml`](./configs/hpg.yaml) points at the copy on `/cmsuf`, which is world-readable:

  ```
  /cmsuf/data/store/user/mmazza/hepattn_clic_data/
  ├── train_clic_fix.root          # 12 GB
  ├── val_clic_fix.root            # 309 MB
  └── test_clic_common_infer.root  # 250 MB
  ```

  The scripts layer that overlay after your config, so `base.yaml` keeps the authors'
  paths and no model config needs editing to run here. If the data moves, change the
  overlay only.

## Submitting a training

Everything is submitted **from the experiment directory**, and the scripts `cd` back to
it on the compute node:

```shell
cd /path/to/hepattn/src/hepattn/experiments/clic
mkdir -p slurm_logs        # sbatch refuses to start if the --output directory is missing
```

One script per kind of job. The model config only defines the model; each script sets the
hardware-dependent settings (device count, batch size per GPU, matcher) on the `main.py`
command line and layers `configs/hpg.yaml` for the data paths, so any config in `configs/`
runs on either machine unchanged.

| Script | Partition | GPUs | `--mem` | Sets | Use |
|---|---|---|---|---|---|
| `submit_training_b200.sh` | `hpg-b200` | 1 B200 | 60G | batch 2048, `configs/matcher_jv.yaml` | fastest way to train |
| `submit_training_l4.sh` | `hpg-turin` | 3 L4 | 150G | batch 170/GPU x 2 accumulation (the paper's global 1020), `configs/matcher_lap1015.yaml` | the paper's geometry |
| `submit_eval_l4.sh` | `hpg-turin` | 1 L4 | 50G | inference mode via `configs/eval.yaml` | evaluate a checkpoint, minutes |
| `submit_validate_run.sh` | `hpg-turin` | 1 L4 | 50G | | re-score a checkpoint on the validation set |

All take the config as their first argument, and anything after it passes straight through
to `main.py`:

```shell
sbatch submit_training_b200.sh configs/base_small.yaml --name my_run
sbatch submit_training_l4.sh configs/base_small.yaml --name my_run
sbatch --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> submit_eval_l4.sh
```

The B200 script needs the GPU matcher's extension built once with
`setup/build_torch_linear_assignment.sh`; it checks for it before training starts. The L4
script needs the GIL-releasing `lap1015` build (`python -c "import lap1015; print(lap1015.releases_gil)"`).

Add `--mail-type=END,FAIL --mail-user=<you>` to the `sbatch` line for e-mail notifications.

### Preflight first, and chain the full run behind it

Before committing a multi-hour allocation, run the same launch path for a few hundred steps
and chain the full run behind it, so it starts only if the preflight passes:

```shell
pf=$(sbatch --parsable --time=00:30:00 --job-name=pf submit_training_b200.sh configs/base_small.yaml \
       --name pf_my_run --trainer.max_steps=300)
sbatch --dependency=afterok:$pf submit_training_b200.sh configs/base_small.yaml --name my_run
```

Check the preflight's SLURM log for the parameter count in `ModelSummary`; a parameter count
alone instantiates every layer but never runs a forward pass, so only a preflight can see a
shape error inside an attention kernel.

### Key rule: devices must match the allocation

`--ntasks-per-node` (SBATCH) must equal `--trainer.devices`, and `--nodes` must equal
`--trainer.num_nodes`. Each script sets both consistently; if you change the GPU count,
change both. Global batch = `num_nodes x devices x data.batch_size x accumulate_grad_batches`.
The learning rate is not batch-scaled, so keep the global batch at the paper's 1020 unless
you mean to change it.

### What the launcher does

Multi-GPU scripts start one task per GPU through [`run_task.sh`](./run_task.sh), which gives
each rank a private Triton/Inductor compile cache on node-local disk before starting the
container. Without it, the ranks race on the shared `$HOME/.triton` cache when the `Compile`
callback compiles the model, one dies with `Text file busy`, and the job hangs on NCCL.

`COMET_MODE=offline` is exported because the compute nodes have no internet and no
`COMET_API_KEY`. The Comet archive lands in the run folder; the CLI also attaches a
`CSVLogger`, so train and val losses are always in `logs/<run>/csv_metrics/metrics.csv`
regardless of Comet.

## Measured runtimes: what to request

Every `#SBATCH --time` in the submit scripts is set from a measured run, at 1.3x the
measured wall time rounded up to the hour, and carries a comment naming that run. Do not
copy a `--time` line from another script. The measurements on this code, all 200 epochs:

| model | hardware | matcher | wall time | request | job |
|---|---|---|---|---|---|
| paper model, 12.1M (`base.yaml`) | 6x L4 (2 nodes), batch 170/GPU | scipy | 19 h 05 | 25 h | 37233919 (paper clone) |
| small, 0.82M (`base_small.yaml`) | 3x L4, batch 170/GPU x 2 accumulation | scipy | 20 h 02 | 27 h | 39236741 (paper clone) |
| small | 1x B200, batch 2048 | `device_solver: jv` | **8 h 59** at 311 ms/step | 12 h | 41758027 |
| small | 1x B200, batch 2048 | host `lap1015_late` | 1,591 ms/step over 300 steps, 84% in the host solve (projected 45 h) | do not use on a B200 | 41755775 |
| small | 3x L4, batch 170/GPU x 2 accumulation | `lap1015_late` | 15 h 07 | 20 h | 41750149 |

For orientation only, the head-based v7 model (0.70M) at the B200 geometry took 6 h 28 to
7 h 40 with the GPU matcher and 23 h 15 with the host matcher (`main`, README_HPG.md there).
Expect this code to stay about a quarter slower per step than those numbers at the same
geometry, and budget for it: 311 ms/step here against 251-255 ms/step for the head model.
**The difference is the GPU matcher's kernel, not the model code.** Profiled at the trained
state (resumed from the epoch-199 checkpoints, jobs 42005058/42005059), the Jonker-Volgenant
kernel takes 84 ms per step on this model's cost matrices against 47 ms on head's; every
other kernel class agrees. The kernel runs one assignment problem per thread and its
Dijkstra loop count is set by the cost matrix, so its time depends on the trained model --
it is 170 ms per step on a fresh model, falls over the first 40 epochs (the per-epoch time
curve every run shows), and settles higher for this model than for head's. That is also
why a 300-step pre-flight cannot see the difference: fresh models of both codes step at
360 ms. The obvious model-side candidate, `Dense`'s gated SwiGLU, was tested and does not
move the step (jobs 41992197 / 41992198, `studies/swiglu_silu/`); the environment does not
either. Before 2026-09-13 `MatcherTimer` reported the device solver at ~0.5% of the step;
that was the launch time of an asynchronous kernel, fixed the same day. See
`studies/step_time_gap/`. Separate from all of that, the paper's `Compile` callback compiled
the whole model as one graph and stepped a further 2.2x slower on a B200 until it was
replaced by head's encoder/decoder compile; a run that steps near 790 ms is hitting that,
not this.

**No measurement for your case?** Run a preflight and project. Submit the training script
with a short step cap and a short limit, then read the projection off its log:

```shell
sbatch --time=00:30:00 submit_training_b200.sh <cfg> --trainer.max_steps=300
python project_runtime.py slurm_logs/slurm-<jobid>.<name>.out --epochs 200
```

It fits the step rate after torch.compile warm-up and prints the projected wall time and the
`--time` to request. It errs on the long side (epoch 0 runs slower than steady state), and it
only reports: the person submitting sets the limit on the full run.

## Outputs & monitoring

- **Run folder:** `logs/<name>_<YYYYMMDD>-T<HHMMSS>/` with `ckpts/`, the resolved
  `config.yaml`, `csv_metrics/`, and the Comet offline archive. `<name>` is the
  config's `name:` field unless overridden with `--name`.
- **SLURM stdout/stderr:** `slurm_logs/slurm-<jobid>.<jobname>.out`.

```shell
squeue -u $USER                                        # queued/running jobs
tail -f slurm_logs/slurm-<jobid>.*.out                 # live training log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it finishes
scancel <jobid>                                        # cancel
```

## Evaluation and validation

All on one L4. Each layers [`configs/eval.yaml`](./configs/eval.yaml) over the run's
config, which applies the evaluation rules from [`README.md`](./README.md): fp32,
`torch` attention, inference-mode data, and no `Compile` callback.

| Script | Purpose |
|---|---|
| `submit_eval_l4.sh` | Parameterised eval: `RUN_DIR` and `CKPT_NAME` from the environment. Writes `<ckpt>__test.root` next to the checkpoint. |
| `submit_validate_run.sh` | Validation-only pass: re-score a checkpoint on the val set under the current code, into a fresh `logs/_val_<jobid>/`. Optional `CONFIG` scores it under another run's objective. |

```shell
sbatch --job-name=clic-eval-paper \
       --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<epoch=...ckpt> \
       submit_eval_l4.sh
```

Produce the performance plots with
[`notebooks/performance.ipynb`](./notebooks/performance.ipynb). To compare with the
paper's figures use the `mpflow_proxy` output, which is what the paper plots.
