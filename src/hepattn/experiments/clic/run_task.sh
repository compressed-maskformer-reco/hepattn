#!/bin/bash
# Per-rank launcher: `srun ./run_task.sh <command...>` runs <command> once per rank with a
# private compile cache.
#
# The Compile callback puts the model through torch.compile -> inductor -> triton. Triton caches
# compiled kernels in $HOME/.triton/cache and installs them with an atomic rename. With several
# ranks compiling the same kernels concurrently against one shared NFS home, that rename races
# and a rank dies with `OSError: [Errno 26] Text file busy`, which then hangs the whole job on
# NCCL. Give every rank its own compile cache on node-local disk instead. The variables are
# inherited by whatever runs next, including a process inside an apptainer container.
export TRITON_CACHE_DIR="/var/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
export TORCHINDUCTOR_CACHE_DIR="/var/tmp/inductor_cache_${SLURM_JOB_ID}_${SLURM_PROCID}"
export TMPDIR=/var/tmp
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

exec "$@"
