#!/bin/bash
# Parameterized CLIC validation-only pass: re-score an existing checkpoint on the val set
# under the *current* code, so two runs trained at different times / precisions / batch
# sizes can be compared on loss-independent quality metrics (eff, pur, mask purity/recall,
# regression residuals) with no confounds.
#
# Setting CONFIG to a *different* run's config scores the checkpoint under that run's
# objective, which puts two models on one yardstick.
#
# Everything is written to a fresh logs/_val_<jobid>/ so Lightning's SaveConfigCallback
# never collides with an existing config.yaml (it aborts rather than overwrite).
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
cd "$CLIC"
export TMPDIR=/var/tmp/

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"
CONFIG_IN="${CONFIG:-${RUN_DIR}/config.yaml}"

# Fresh output dir; redirect both default_root_dir and the Comet offline dir into it so
# nothing is written back into the run folder being scored.
OUTDIR="${CLIC}/logs/_val_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
CONFIG_RUN="${OUTDIR}/config_in.yaml"
python3 - "$CONFIG_IN" "$CONFIG_RUN" "$OUTDIR" <<'PYEOF'
import re, sys
src, dst, outdir = sys.argv[1:4]
txt = open(src).read()
txt = re.sub(r'(offline_directory|default_root_dir|save_dir):\s*\S+', lambda m: f"{m.group(1)}: {outdir}", txt)
open(dst, "w").write(txt)
PYEOF

echo "Hostname: $(hostname)"
echo "RUN_DIR:  ${RUN_DIR}"
echo "CKPT:     ${CKPT}"
echo "CONFIG:   ${CONFIG_IN}  ->  ${CONFIG_RUN}"
echo "OUTDIR:   ${OUTDIR}"

PYTORCH_CMD="python main.py validate \
  --config ${CONFIG_RUN} \
  --config configs/hpg.yaml \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  --ckpt_path $CKPT"

apptainer run --nv --bind /blue/,/cmsuf/ \
  "$REPO/pixi.sif" pixi run -e clic $PYTORCH_CMD

echo "Done!"
