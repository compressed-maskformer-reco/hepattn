#!/bin/bash
# Validation-only pass for CLIC: re-score an existing checkpoint on the val set under the
# current code, so runs trained at different times, precisions or batch sizes can be compared
# on loss-independent quality metrics (efficiency, purity, mask purity/recall, regression
# residuals) with no confounds. Setting CONFIG to a different run's config scores the
# checkpoint under that run's objective instead.
#
# Everything is written to a fresh logs/_val_<jobid>/ so Lightning's SaveConfigCallback never
# collides with an existing config.yaml (it aborts rather than overwrite).
#
# Submit with:
#   sbatch --job-name=clic-val-<tag> \
#          --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file>[,CONFIG=<cfg>] \
#          submit_validate_run.sh

#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/slurm-%j.%x.out

set -euo pipefail

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
cd "$CLIC"
export TMPDIR=/var/tmp/

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"
CONFIG_IN="${CONFIG:-${RUN_DIR}/config.yaml}"

# Fresh output dir; redirect both default_root_dir and the Comet offline dir into it so
# nothing is written back into the run folder being scored.
OUTDIR="${CLIC}/logs/_val_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
CONFIG_RUN="${OUTDIR}/config_in.yaml"
python3 - "$CONFIG_IN" "$CONFIG_RUN" "$OUTDIR" <<'PY'
import re, sys
src, dst, outdir = sys.argv[1:4]
txt = open(src).read()
txt = re.sub(r'(offline_directory|default_root_dir):\s*\S+', lambda m: f"{m.group(1)}: {outdir}", txt)
open(dst, "w").write(txt)
PY

echo "Hostname: $(hostname)"
echo "RUN_DIR:  ${RUN_DIR}"
echo "CKPT:     ${CKPT}"
echo "CONFIG:   ${CONFIG_IN}  ->  ${CONFIG_RUN}"
echo "OUTDIR:   ${OUTDIR}"

PYTORCH_CMD="python main.py validate \
  --config ${CONFIG_RUN} \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  $DATA_ARGS \
  --ckpt_path $CKPT"

apptainer run --nv --bind /blue/,/cmsuf/ \
  "$REPO/pixi.sif" pixi run -e clic $PYTORCH_CMD

echo "Done!"
