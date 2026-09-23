#!/bin/bash
# Train one CLIC config on 3 L4 (one node) with the host matcher (lap1015, threaded).
# Submit FROM this directory:
#   sbatch submit_training_l4.sh <config.yaml> [extra main.py args...]
# Extra arguments pass straight through to main.py, e.g. --name my_run or --trainer.max_epochs=2.

#SBATCH --job-name=clic-train-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
# --time: 3x L4, batch 170 x 2 accumulation, host matcher, 200 epochs of the paper's small model:
# measured 20 h 02 (job 39236741, scipy solver, paper clone), 15 h 07 with lap1015_late on this
# code (job 41750149). 1.3x the slower of the two, rounded up to the hour, because this script
# also launches runs that keep the scipy solver. The full paper model on 6 L4 took 19 h 05
# (job 37233919). See README_HPG.md, Measured runtimes.
#SBATCH --time=27:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=slurm_logs/slurm-%j.%x.out

# Resolve the repository from wherever this script was submitted, so the job runs against
# the clone it was launched from instead of one person's checkout. sbatch sets
# SLURM_SUBMIT_DIR to the directory it was submitted from, which these scripts document as
# this one; the fallback keeps the script usable when run directly.
CLIC="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$CLIC/../../../.." && pwd)"

CONFIG_PATH="${1:?Usage: sbatch submit_training_l4.sh <config.yaml> [extra args]}"
shift
echo "Using config: $CONFIG_PATH"

# Load CUDA matching the container build
module load cuda/12.8.1

# Compute nodes have no internet / COMET_API_KEY: run the CometLogger offline.
export COMET_MODE=offline

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
echo "started: $(date -Is)"
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd "$CLIC"
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Make the GPU matcher (configs/matcher_jv.yaml) available when its out-of-environment build
# exists: apptainer forwards APPTAINERENV_* into the container. Harmless when unused.
TLA_DIR="$REPO/vendor/torch-linear-assignment"
if [ -d "$TLA_DIR" ]; then
  export APPTAINERENV_PYTHONPATH="$TLA_DIR"
  export APPTAINERENV_LD_LIBRARY_PATH="$REPO/.pixi/envs/clic/lib"
fi

# configs/hpg.yaml layers the HPG data paths over the model config, which keeps the
# authors' paths. batch_size 170/GPU x 3 L4 x 2 accumulation steps = global 1020, the paper's global batch with the same per-rank shape as the 6-GPU script (the loss is normalised per rank, so this averages the same way as 6 ranks of 170).
# Override any of these through the extra arguments.
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH --config configs/hpg.yaml --config configs/matcher_lap1015.yaml \
    --trainer.devices=3 --trainer.num_nodes=1 --data.batch_size=170 \
    --trainer.accumulate_grad_batches=2 $*"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image.
# srun in front for multi-GPU DDP; run_task.sh gives each rank a private compile cache.
APPTAINER_CMD="srun ./run_task.sh apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "finished: $(date -Is)"
echo "Done!"
