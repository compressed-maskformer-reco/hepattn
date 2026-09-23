#!/bin/bash
# Parameterized CLIC eval: run inference from a run's checkpoint to produce
# <ckpt>__test.h5 and (via PflowPredictionWriter) <ckpt>__test.root next to the ckpt.
#
# Submit with:
#   sbatch --job-name=clic-eval-<tag> \
#          --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> \
#          submit_eval_l4.sh
#
# RUN_NAME instead of RUN_DIR resolves the newest logs/<RUN_NAME>_* at run time, which is what
# lets an evaluation be queued behind a training that has not started yet: the run directory
# carries a timestamp fixed when the training begins, so it cannot be named at submit time.
#   sbatch --dependency=afterok:<train job> --job-name=clic-eval-<tag> \
#          --export=ALL,RUN_NAME=<the config's name:> submit_eval_l4.sh
#
# CKPT_NAME defaults to the most recently written checkpoint in the run, which for a completed
# training is its last epoch. Name it explicitly to evaluate any other one.
#
# EVAL_CONFIG overrides the evaluation overlay, for models configs/eval.yaml cannot describe:
# a Linformer run needs studies/linformer/configs/eval_linformer.yaml, whose comment says why.

#SBATCH -p hpg-turin
#SBATCH --account=your-account
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
# --time: 1x L4 evaluation of the small model: under 2 min on head code (jobs 41332603-06). 30 min.
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=slurm_logs/slurm-%j.%x.out

EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"

module load cuda/12.8.1
export COMET_MODE=offline

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "RUN_NAME: ${RUN_NAME:-<unset, RUN_DIR given>}"
nvidia-smi

# Resolve the repository from wherever this script was submitted, so the job runs against
# the clone it was launched from instead of one person's checkout. sbatch sets
# SLURM_SUBMIT_DIR to the directory it was submitted from, which these scripts document as
# this one; the fallback keeps the script usable when run directly.
CLIC="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$CLIC/../../../.." && pwd)"

cd "$CLIC"
echo "Working dir: ${PWD}"

# Resolve the run directory here rather than at submit time, so this can be queued behind a
# training whose timestamped directory does not exist yet.
if [ -z "${RUN_DIR:-}" ]; then
  : "${RUN_NAME:?export RUN_DIR=logs/<run_folder> or RUN_NAME=<the name: field of the config>}"
  RUN_DIR=$(ls -1d "logs/${RUN_NAME}"_* 2>/dev/null | sort | tail -1)
  [ -n "${RUN_DIR}" ] || { echo "no run directory matches logs/${RUN_NAME}_*"; exit 1; }
fi
[ -d "${RUN_DIR}" ] || { echo "no such run directory: ${RUN_DIR}"; exit 1; }

# Newest by modification time, not by name: "epoch=99" sorts after "epoch=199" lexically.
if [ -z "${CKPT_NAME:-}" ]; then
  CKPT_NAME=$(ls -1t "${RUN_DIR}"/ckpts/*.ckpt 2>/dev/null | head -1 | xargs -r basename)
  [ -n "${CKPT_NAME}" ] || { echo "no checkpoints in ${RUN_DIR}/ckpts"; exit 1; }
fi

export TMPDIR=/var/tmp/

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"

echo "RUN_DIR: ${RUN_DIR}"
echo "CKPT_NAME: ${CKPT_NAME}"
echo "EVAL_CONFIG: ${EVAL_CONFIG}"
[ -f "${EVAL_CONFIG}" ] || { echo "no such eval config: ${EVAL_CONFIG}"; exit 1; }

# The eval overlay applies the README's evaluation rules (fp32, torch attention, inference data).
PYTORCH_CMD="python main.py test \
  --config ${RUN_DIR}/config.yaml \
  --config configs/hpg.yaml \
  --config ${EVAL_CONFIG} \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  --ckpt_path $CKPT"

PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

APPTAINER_CMD="apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif $PIXI_CMD"

echo "Running: $PYTORCH_CMD"
$APPTAINER_CMD
echo "Done!"
