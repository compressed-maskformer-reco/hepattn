#!/usr/bin/env bash
# Evaluate a trained Linformer checkpoint.
#   CKPT=/abs/path/to.ckpt bash polaris/submit_linformer_eval.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/polaris/env.sh"
: "${CKPT:?set CKPT=/abs/path/to/checkpoint.ckpt}"
# One log per run dir, so evaluating two arms back to back does not blend their logs.
RUN_NAME="$(basename "$(dirname "$(dirname "${CKPT}")")")"
NAME="clic-eval-${RUN_NAME:0:40}"
mkdir -p "${REPO_DIR}/polaris/logs"
: > "${REPO_DIR}/polaris/logs/${NAME}.log"
set -x
qsub -A "${PBS_PROJECT}" \
     -q "${QUEUE_DEBUG:-debug}" \
     -l "filesystems=${PBS_FILESYSTEMS}" \
     -l "walltime=${EVAL_WALLTIME:-01:00:00}" \
     -N "${NAME}" \
     -o "${REPO_DIR}/polaris/logs/${NAME}.log" \
     -v "REPO_DIR=${REPO_DIR},CKPT=${CKPT}" \
     "${REPO_DIR}/polaris/linformer_eval.pbs"
