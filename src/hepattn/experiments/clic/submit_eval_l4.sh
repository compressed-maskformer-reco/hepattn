#!/bin/bash
# Parameterized CLIC eval: run inference from a run's best checkpoint to produce
# <ckpt>__test.h5 and (via PflowPredictionWriter) <ckpt>__test.root next to the ckpt.
#
# Submit with:
#   sbatch --job-name=clic-eval-<tag> \
#          --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> \
#          submit_eval_l4.sh

#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
# --time: 1x L4 evaluation: measured under 2 min (jobs 41332603-06). 1.3x, rounded up to the hour. See README_HPG.md, Measured runtimes.
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "RUN_DIR: ${RUN_DIR}"
echo "CKPT_NAME: ${CKPT_NAME}"
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

cd "$CLIC"
echo "Working dir: ${PWD}"

export TMPDIR=/var/tmp/

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"

PYTORCH_CMD="python main.py test \
  --config ${RUN_DIR}/config.yaml \
  --config configs/eval.yaml \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  $DATA_ARGS \
  --ckpt_path $CKPT"

PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

APPTAINER_CMD="apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"

echo "Running: $PYTORCH_CMD"
$APPTAINER_CMD
echo "Done!"
