#!/bin/bash

#SBATCH --job-name=clic-train-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
# --time: 3x L4, v6 at batch 256: measured 31 h 55 (job 38469247); v7 is faster. 1.3x, rounded up to the hour. See README_HPG.md, Measured runtimes.
#SBATCH --time=42:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_logs/slurm-%j.%x.out

# Load CUDA matching the container build
module load cuda/12.8.1

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi:"
nvidia-smi

# Submitted from this directory (src/hepattn/experiments/clic), which is where the job finds
# the repository and writes its log (slurm_logs/).
CLIC=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "$CLIC/main.py" ] || { echo "ERROR: submit from src/hepattn/experiments/clic, not $CLIC" >&2; exit 1; }
REPO=$(cd "$CLIC/../../../.." && pwd)

# The configs carry another cluster's data paths; on HPG the data is on /cmsuf.
CLIC_DATA=${CLIC_DATA:-/cmsuf/data/store/user/mmazza/hepattn_clic_data}
DATA_ARGS="--data.train_path=$CLIC_DATA/train_clic_fix.root \
  --data.valid_path=$CLIC_DATA/val_clic_fix.root \
  --data.test_path=$CLIC_DATA/test_clic_common_infer.root"

# Move to workdir
cd "$CLIC"
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the training
echo "Running training script..."

# Config is passed as the first argument, e.g.
#   sbatch submit_training_l4.sh studies/linformer_size/configs/small_ref.yaml
CONFIG_PATH="${1:?Usage: sbatch submit_training_l4.sh <config.yaml> [extra main.py args...]}"
EXTRA_ARGS=("${@:2}")
echo "Using config: $CONFIG_PATH"

# Python command that will be run.
# devices=3 on 1 node = 3 L4 GPUs, matching the SBATCH allocation above.
# batch 512 per GPU, the most a 24 GB L4 holds; a later --data.batch_size=N in the extra args wins.
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH --trainer.devices=3 --data.batch_size=512 $DATA_ARGS ${EXTRA_ARGS[*]}"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# srun in front for multi-GPU DDP
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
