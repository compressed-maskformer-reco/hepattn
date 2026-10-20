#!/bin/bash
# Jet- and particle-level physics plots for the decoder linformer_k scan, as a CPU batch job: jet clustering and
# particle matching want cores (compute_jets(n_procs=16)), and the pipeline reads the .root files the evaluation
# jobs wrote. Once a script has cached its pipeline (studies/linformer_dim/cache/), redraws run on a login node.
#
# Takes script paths relative to the clic experiment dir and runs them in order:
#   sbatch studies/linformer_dim/submit_plots.sh studies/linformer_dim/plot_jet_iqr.py studies/linformer_dim/plot_performance.py
# CONVENTIONS (default mpflow_proxy) is passed through, e.g. sbatch --export=ALL,CONVENTIONS=mpflow,mpflow_proxy ...
#
# Submitted from src/hepattn/experiments/clic, like the other submit scripts.

#SBATCH --job-name=clic-linformer-dim-plots
#SBATCH -p hpg-default
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# --time / --mem: the paper-tag model-size study's equivalent job (two scripts, five arms, both conventions) took
# 8 min 43 s and 40.2 GB peak on 16 cores (job 42119839); 1.3x rounded up to the hour, memory at 3x the peak.
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/slurm-%j.%x.out

set -euo pipefail

CLIC=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "$CLIC/main.py" ] || { echo "ERROR: submit from src/hepattn/experiments/clic, not $CLIC" >&2; exit 1; }
cd "$CLIC"
export TMPDIR=/var/tmp/
export CONVENTIONS="${CONVENTIONS:-mpflow_proxy}"

echo "Hostname: $(hostname)   cores: $(nproc)   CONVENTIONS: ${CONVENTIONS}"
for s in "$@"; do
  echo "================================================================ $s"
  pixi run -e clic python "$s"
done
echo "Done!"
