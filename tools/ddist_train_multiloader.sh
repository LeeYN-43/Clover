#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
HOSTS=$3
PORT=$4

echo "The distributed PORT is ${PORT}"

export OMP_NUM_THREADS=8



torchrun --nproc_per_node=$GPUS --master_port=$PORT \
  --master_addr $HOSTS --node_rank=$5 --nnodes=$6 \
  tools/train_multiloader.py $CONFIG --launcher pytorch ${@:7}
