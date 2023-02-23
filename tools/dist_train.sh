#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

#pwd; ls -l  > dir_pwd.txt
#ls -l $(dirname "$0") > dir_0.txt
source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate cvproj

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
