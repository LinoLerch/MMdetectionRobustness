#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CHECKPOINT=$4
OUT=$5
TIME=$6
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:7}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
sbatch -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:1 \
    --time=${TIME} \
    ${SRUN_ARGS} \
    ./tools/robust_benchmark.sh ${CONFIG} ${CHECKPOINT} ${OUT}