# Running CLIC on HiPerGator (HPG)

HPG-specific instructions for training and evaluating the CLIC particle-flow model
on UF's HiPerGator cluster. For the model/paper overview and data-format details see
[`README.md`](./README.md).

## Prerequisites

- **GPU partitions:**
  - `hpg-b200` — B200 192 GB (up to 4/node)
  - `hpg-turin` — L4 24 GB (up to 3/node)
- **Container + env:** the repo ships a pixi Apptainer image at the repo root
  (`pixi.sif`). Training runs inside it via `apptainer run --nv ... pixi run -e clic ...`;
  the submit scripts already do this — you don't need to enter it by hand to submit.
- **CUDA module:** `module load cuda/12.8.1` (matches the container build; already in
  the submit scripts).

## Data

The CLIC ROOT files can live anywhere the compute nodes can read. On UF HiPerGator we
keep them on the CMS store (`/cmsuf`) rather than `/blue`, to stay under the `/blue`
quota (the submit scripts bind both `/blue` and `/cmsuf` into the container):

```
/cmsuf/data/store/user/<user>/hepattn_clic_data/
├── train_clic_fix.root          # 12 GB
├── val_clic_fix.root            # 309 MB
└── test_clic_common_infer.root  # 250 MB
```

Configs point at these paths directly (`data.train_path` / `valid_path` / `test_path`).
If you write a new config, make sure its paths point here and **not** at a Jupyter/
container path like `/home/jovyan/...`, which does not exist on the compute nodes.

## Submitting a training

All commands run from the experiment directory:

```shell
cd /path/to/hepattn/src/hepattn/experiments/clic
```

### Submit scripts

One script per kind of job. The model config only defines the model; each script sets the
hardware-dependent settings (device count, batch size per GPU, matcher) on the `main.py`
command line, so any config in `configs/` runs on either machine unchanged.

| Script | Partition | GPUs | `--mem` | Sets | Use |
|---|---|---|---|---|---|
| `submit_training_b200.sh` | `hpg-b200` | 1 B200 | 60G | batch 2048, `device_solver: jv` | fastest way to train |
| `submit_training_l4.sh` | `hpg-turin` | 3 L4 | 150G | batch 512/GPU, host matcher | the GPU matcher does not pay on L4s |
| `submit_eval_l4.sh` | `hpg-turin` | 1 L4 | 50G | inference mode via `configs/eval.yaml` | evaluate a checkpoint |
| `submit_validate_run.sh` | `hpg-turin` | 1 L4 | 50G | | re-score a checkpoint on the validation set |

The training scripts take the config as their first argument, and anything after it passes
straight through to `main.py`:

```shell
sbatch submit_training_b200.sh configs/base.yaml
sbatch submit_training_l4.sh configs/base.yaml
sbatch submit_training_l4.sh configs/base.yaml --data.batch_size=256   # if a large model OOMs on 24 GB
sbatch --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> submit_eval_l4.sh
```

Before the first submission, replace the placeholders in each script (`/path/to/hepattn`,
`your-account`, `your-email@example.com`), or override the `#SBATCH` ones on the `sbatch`
command line.

The B200 script needs the GPU matcher's extension, built once into
`vendor/torch-linear-assignment` (see [below](#solving-the-matching-on-the-gpu)); it checks
for it before training starts.

### Key rule: devices must match the allocation

`--ntasks-per-node` (SBATCH) **must equal** `--trainer.devices`, and
`--nodes` **must equal** `--trainer.num_nodes`. The scripts set the `trainer.*`
overrides on the `python main.py` line to match their SBATCH directives — if you
change the GPU count, change both. Total GPUs = `num_nodes × devices`; global batch =
`num_nodes × devices × data.batch_size`, and the scripts set `data.batch_size` for their
hardware (2048 on a B200, 512 on an L4).

### Interactive (debug) run

```shell
apptainer shell --nv --bind /blue/ /path/to/hepattn/pixi.sif
pixi shell -e clic
python main.py fit --config configs/base.yaml --trainer.devices=1
```

Everything CLIC on HPG uses the `clic` environment: the submit scripts run
`pixi run -e clic`, and it is the environment to use for the performance notebooks
too. It is the GPU `default` environment plus the analysis packages, so there is no
need to install `default` as well (each environment is ~15 GB).

### Solving the matching on the GPU

A single B200 is not saturated by the CLIC model, so the training step there is
host-bound and the Hungarian matching (device-to-host copy plus a threaded solve)
dominates it: with the host matcher a B200 training takes about 3x longer. The `Matcher`
has an opt-in GPU solver for exactly that case, and `submit_training_b200.sh` turns it on
(`device_solver=jv`) and puts the extension on the container's `PYTHONPATH`. Build it once,
from the repo root, for L4 and B200:

```shell
TORCH_CUDA_ARCH_LIST="8.9;10.0" pixi run -e clic bash setup/build_torch_linear_assignment.sh
```

Set `TORCH_CUDA_ARCH_LIST` explicitly: otherwise the `clic` environment's CUDA activation
sets a list containing `10.1`, which this torch does not accept. See the
[top-level README](../../../../README.md#optional-solving-the-matching-on-the-gpu).
Leave it off on the L4 nodes, which the model already keeps busy.

## Measured runtimes: what to request

Every `#SBATCH --time` in the submit scripts is set from a measured run, at 1.3x the
measured wall time rounded up to the hour, and carries a comment naming that run. Do not
copy a `--time` line from another script: for months every B200 training asked for four
days and finished in seven hours. The reference measurements, all 200 epochs (v6 is the
`base.yaml` model; v7 a smaller variant):

| model | hardware | matcher | wall time | request | job |
|---|---|---|---|---|---|
| v7 | 1x B200, batch 2048 | `device_solver: jv` | 6 h 28 to 7 h 40 | 10 h | 40405423, 41331284 |
| v7 | 1x B200, batch 2048 | host (`lap1015_late`) | 23 h 15 | 31 h | 40400036 |
| v6 | 1x B200, batch 2048 | `device_solver: jv` | 11 h 22 | 15 h | 40393482 |
| v6 | 1x B200, batch 2048 | host | 33 h 03 | 44 h | 38598204 |
| v6 | 3x L4, batch 256/GPU | host | 31 h 55 | 42 h | 38469247 |
| any | 1x L4 evaluation | | under 2 min | 30 min | 41332603 |

Memory requests (60 GB per B200 job, 150 GB for 3x L4, 50 GB for evaluation) have never
been exceeded. The accounting for the apptainer step double-counts the dataloader workers'
shared pages, so do not read `sacct` memory figures as usage.

**No measurement for your model?** Run a short preflight and scale up from its step rate:

```shell
sbatch --time=00:30:00 submit_training_b200.sh configs/<cfg>.yaml --trainer.max_steps=300
```

## Outputs & monitoring

- **Training output folder:** `logs/<config-name>_<YYYYMMDD>-T<HHMMSS>/` (checkpoints in
  `ckpts/`, resolved `config.yaml`, `metadata.yaml`). The `<config-name>` is the
  config's `name:` field (e.g. `base`).
- **SLURM stdout/stderr:** `slurm_logs/slurm-<jobid>.<jobname>.out`.

```shell
squeue -u $USER                          # your queued/running jobs
tail -f slurm_logs/slurm-<jobid>.*.out     # live training log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it finishes
scancel <jobid>                            # cancel
```

## Evaluation

Run on a single GPU (see [`README.md`](./README.md) for the full flag rationale). Note
the required overrides — inference data mode, fp32, and switching attention to `torch`:

```shell
python main.py test \
    --config logs/<run-folder>/config.yaml \
    --data.test_path /cmsuf/data/store/user/<user>/hepattn_clic_data/test_clic_common_infer.root \
    --data.is_inference true \
    --trainer.precision 32-true \
    --matmul_precision highest
```

Also remove the `Compile` callback and set attention to `torch` in the config before
testing. `submit_eval_l4.sh` does all of that through `configs/eval.yaml`; pass the run
directory and checkpoint name as shown above. Produce performance plots with
[`notebooks/performance.ipynb`](./notebooks/performance.ipynb).
