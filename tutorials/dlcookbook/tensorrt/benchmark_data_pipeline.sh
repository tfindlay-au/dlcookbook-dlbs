#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../../scripts/environment.sh
dlbs=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
loglevel=warning

#------------------------------------------------------------------------------#
# Run multi-GPU inference with real data.
#------------------------------------------------------------------------------#
rm -rf ./logs/data_pipeline
python $dlbs run \
       --log-level=$loglevel\
       -Pexp.replica_batch=128\
       -Pexp.num_warmup_batches=50\
       -Pexp.num_batches=200\
       -Pexp.data_dir='"/home/serebrya/data/"'\
       -Vtensorrt.num_prefetchers='[1, 2, 4, 8, 1]'\
       -Vtensorrt.num_decoders='[1, 2, 4, 8, 16]'\
       -Ptensorrt.prefetch_queue_size='"$(${tensorrt.num_prefetchers}*4)$"'\
       -Ptensorrt.inference_queue_size='"$(${tensorrt.num_decoders}*4)$"'\
       -Ptensorrt.fake_inference=true\
       -Pexp.log_file='"${BENCH_ROOT}/logs/data_pipeline/p${tensorrt.num_prefetchers}_d${tensorrt.num_decoders}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.replica_batch,results.throughput,tensorrt.num_prefetchers,tensorrt.num_decoders"
params="$params,tensorrt.prefetch_queue_size,tensorrt.inference_queue_size"
python $parser ./logs/data_pipeline/*.log --output_params ${params}
