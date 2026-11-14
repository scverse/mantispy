#!/bin/bash
# sbatch --account=icb --partition=cpu_p --qos=cpu_normal -c 32 --mem=MEM -t 08:00:00 run_bench.sh DATASET MODE IMPL
set -uo pipefail
W=/lustre/groups/ml01/workspace/tim.treis
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 UV_CACHE_DIR=$W/uv-cache
export NUMBA_CACHE_DIR=$W/perf-map-nb-out/numba_cache NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
cd $W/perf-map-nb
PY=$(hatch env find hatch-test.py3.14-stable)/bin/python
echo "== $1 $2 $3 on $(hostname), $(nproc) cpus allotted"
$PY scratch/map_numba/bench.py "$1" "$2" "$3" $W/perf-map-nb-out/results "${@:4}"
