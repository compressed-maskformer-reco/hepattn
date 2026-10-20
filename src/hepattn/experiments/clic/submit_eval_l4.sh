#!/bin/bash
# Parameterized CLIC eval: run inference from a run's best checkpoint to produce
# <ckpt>__test.h5 and (via PflowPredictionWriter) <ckpt>__test.root next to the ckpt.
#
# Submit with:
#   sbatch --job-name=clic-eval-<tag> \
#          --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> \
#          submit_eval_l4.sh
# or, chained after a training job (run folder and best checkpoint found when the job starts):
#   sbatch --job-name=clic-eval-<tag> --dependency=afterok:<train_job> \
#          --export=ALL,TRAIN_JOB=<train_job> submit_eval_l4.sh
# Add EVAL_CONFIG=configs/eval_linformer.yaml to either for a run with linformer attention.

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

# Chained after a training job, the run folder and best checkpoint are not known at submit time:
#   sbatch --dependency=afterok:<train_job> --export=ALL,TRAIN_JOB=<train_job> submit_eval_l4.sh
# finds the folder whose metadata.yaml names that job, and its checkpoint with the lowest val loss.
if [ -n "${TRAIN_JOB:-}" ] && [ -z "${RUN_DIR:-}" ]; then
  META=$(grep -lx "slurm_job_id: SLURM_${TRAIN_JOB}" logs/*/metadata.yaml | head -1)
  [ -n "$META" ] || { echo "ERROR: no run folder in logs/ for training job $TRAIN_JOB" >&2; exit 1; }
  RUN_DIR=$(dirname "$META")
fi
if [ -z "${CKPT_NAME:-}" ]; then
  CKPT_NAME=$(ls "${RUN_DIR}/ckpts" | sed -n 's/.*val_loss=\([0-9.]*\)\.ckpt$/\1 &/p' | sort -g | head -1 | cut -d' ' -f2)
  [ -n "$CKPT_NAME" ] || { echo "ERROR: no checkpoint in ${RUN_DIR}/ckpts" >&2; exit 1; }
fi
echo "Resolved RUN_DIR: ${RUN_DIR}, CKPT_NAME: ${CKPT_NAME}"

export TMPDIR=/var/tmp/

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"

# EVAL_CONFIG overrides the evaluation overlay, for models configs/eval.yaml cannot describe:
# a run with linformer attention needs configs/eval_linformer.yaml, whose comment says why.
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval.yaml}

PYTORCH_CMD="python main.py test \
  --config ${RUN_DIR}/config.yaml \
  --config ${EVAL_CONFIG} \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  $DATA_ARGS \
  --ckpt_path $CKPT"

PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

APPTAINER_CMD="apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"

echo "Running: $PYTORCH_CMD"
$APPTAINER_CMD || { echo "ERROR: evaluation failed" >&2; exit 1; }
echo "Done!"
