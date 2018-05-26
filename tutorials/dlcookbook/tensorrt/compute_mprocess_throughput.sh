#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache

. ${BENCH_ROOT}/../../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/logparser.py

logdir=./logs
tmpfile=./.tmp/results.json

rm $tmpfile

params="exp.replica_batch,results.throughput,results.time_data,results.mgpu_effective_throughput,exp.num_gpus"
python $script ${logdir} --recursive --output_params ${params} --output_file $tmpfile

python ./compute_mprocess_throughput.py
