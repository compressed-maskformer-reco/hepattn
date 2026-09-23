#!/bin/bash
# Train one CLIC config on a single B200 with the GPU (Jonker-Volgenant) matcher.
#
# The model config carries only the model; the hardware-dependent settings are set here:
# one device, batch 2048 per GPU (a B200 holds it), and the GPU matcher via
# configs/matcher_jv.yaml, which needs the torch-linear-assignment build from
# setup/build_torch_linear_assignment.sh at the path exported below.
#
# Usage, from this directory (mkdir -p slurm_logs first):
#   sbatch submit_training_b200.sh <config.yaml> [extra main.py args...]
#   sbatch submit_training_b200.sh configs/base_small.yaml --name my_run
#   sbatch --time=00:30:00 submit_training_b200.sh configs/base_small.yaml --trainer.max_steps=300   # preflight

#SBATCH --job-name=clic-train-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
# --time: 1x B200, jv matcher, 200 epochs of the paper's small model: 8 h 59 measured
# (job 41758027, 2026-09-11, 311 ms/step); 1.3x rounded up to the hour. The 300-step
# preflight this replaces read 367 ms/step and projected 10.4 h, which is the bias
# project_runtime.py warns about.
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1
export COMET_MODE=offline

# Resolve the repository from wherever this script was submitted, so the job runs against
# the clone it was launched from instead of one person's checkout. sbatch sets
# SLURM_SUBMIT_DIR to the directory it was submitted from, which these scripts document as
# this one; the fallback keeps the script usable when run directly.
CLIC="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$CLIC/../../../.." && pwd)"
CONFIG_PATH="${1:?Usage: sbatch submit_training_b200.sh <config.yaml> [extra main.py args...]}"
EXTRA_ARGS=("${@:2}")

# Pin the pixi environment explicitly: sbatch --export=ALL forwards the submitting shell's
# variables, and a submitting `pixi shell` would otherwise choose the environment.
unset PIXI_ENVIRONMENT_NAME PIXI_IN_SHELL PIXI_PROMPT CONDA_PREFIX || true

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
echo "config: $CONFIG_PATH"
nvidia-smi

cd $REPO/src/hepattn/experiments/clic/
echo "Moved dir, now in: ${PWD}"
export TMPDIR=/var/tmp/

# The GPU matcher's extension lives outside the pixi environment (see
# setup/build_torch_linear_assignment.sh); it needs the environment's libstdc++ at run time.
export APPTAINERENV_PYTHONPATH=$REPO/vendor/torch-linear-assignment
export APPTAINERENV_LD_LIBRARY_PATH=$REPO/.pixi/envs/clic/lib

# Fail here, before burning the allocation, if the build is missing or CPU-only.
srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic \
  python -c "from torch_linear_assignment import _backend as b; assert b.has_cuda(), 'CPU-only build'; print('torch-linear-assignment ok, has_cuda: True')"

# Own Inductor/Triton cache: no kernels inherited from another run.
CACHE="/var/tmp/clic_${SLURM_JOB_ID}"
mkdir -p "${CACHE}/inductor" "${CACHE}/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export APPTAINERENV_TRITON_CACHE_DIR="${CACHE}/triton"

echo "started: $(date -Is)"
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH --config configs/hpg.yaml --config configs/matcher_jv.yaml \
  --trainer.devices=1 --trainer.num_nodes=1 \
  --data.batch_size=2048 \
  ${EXTRA_ARGS[*]}"

PIXI_CMD="pixi run -e clic $PYTORCH_CMD"
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "finished: $(date -Is)"
