#!/bin/bash
# Train one CLIC config on a single B200 with the GPU (Jonker-Volgenant) matcher.
#
# The model config carries only the model; the hardware-dependent settings are set here:
# one device, batch 2048 per GPU (a B200 holds it; the config's default fits an L4), and the
# GPU matcher, which is what makes a single B200 worthwhile: with the host matcher the step
# is CPU-bound and takes 3x longer. The matcher needs the torch-linear-assignment build from
# setup/build_torch_linear_assignment.sh at the path exported below.
#
# Usage:
#   sbatch submit_training_b200.sh <config.yaml> [extra main.py args...]
#   sbatch submit_training_b200.sh studies/linformer_size/configs/small_ref.yaml
#   sbatch --time=00:30:00 submit_training_b200.sh studies/linformer_size/configs/small_ref.yaml --trainer.max_steps=300   # preflight

#SBATCH --job-name=clic-train-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
# --time: 1x B200 with the jv matcher, 200 epochs: v7 measured 7 h 40 (job 40405423), v6 11 h 22 (job 40393482). 1.3x the v6 time, rounded up. See README_HPG.md, Measured runtimes.
#SBATCH --time=15:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_logs/slurm-%j.%x.out

set -euo pipefail

# Load CUDA matching the container build
module load cuda/12.8.1

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
CONFIG_PATH="${1:?Usage: sbatch submit_training_b200.sh <config.yaml> [extra main.py args...]}"
EXTRA_ARGS=("${@:2}")

# Comet variables
echo "Setting comet experiment key"
COMET_EXPERIMENT_KEY=$(date +%s)
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo ${COMET_WORKSPACE:-}

# Print host info
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

cd "$CLIC"
echo "Moved dir, now in: ${PWD}"
export TMPDIR=/var/tmp/

# The GPU matcher's extension lives outside the pixi environment (see
# setup/build_torch_linear_assignment.sh); it needs the environment's libstdc++ at run time.
# Prepended to the submitting shell's PYTHONPATH, which would otherwise not reach the container.
export APPTAINERENV_PYTHONPATH=$REPO/vendor/torch-linear-assignment${PYTHONPATH:+:$PYTHONPATH}
export APPTAINERENV_LD_LIBRARY_PATH=$REPO/.pixi/envs/clic/lib

# Fail here, before burning the allocation, if the build is missing or CPU-only.
srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic \
  python -c "from torch_linear_assignment import _backend as b; assert b.has_cuda(), 'CPU-only build'; print('torch-linear-assignment ok, has_cuda: True')"

echo "Running training script..."
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH \
  --trainer.devices=1 \
  --data.batch_size=2048 \
  --model.model.matcher.init_args.device_solver=jv \
  $DATA_ARGS \
  ${EXTRA_ARGS[*]}"

PIXI_CMD="pixi run -e clic $PYTORCH_CMD"
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
