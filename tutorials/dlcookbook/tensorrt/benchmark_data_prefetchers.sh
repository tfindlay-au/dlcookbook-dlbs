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
#-Pexp.data_dir='"/home/serebrya/data/"'\
#------------------------------------------------------------------------------#
rm -rf ./logs/data_prefetchers
python $dlbs run \
       --log-level=$loglevel\
       -Pexp.replica_batch=128\
       -Pexp.num_warmup_batches=4\
       -Pexp.num_batches=10\
       -Pexp.data_dir='"/dev/shm/train/"'\
       -Vtensorrt.num_prefetchers='[1, 2, 4, 8, 16]'\
       -Ptensorrt.prefetch_queue_size='"$(${tensorrt.num_prefetchers}*4)$"'\
       -Ptensorrt.inference_queue_size='"$(${tensorrt.num_prefetchers}*4)$"'\
       -Ptensorrt.fake_decoder=true\
       -Ptensorrt.fake_inference=true\
       -Pexp.log_file='"${BENCH_ROOT}/logs/data_prefetchers/prefetchers_${tensorrt.num_prefetchers}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.replica_batch,results.throughput,tensorrt.num_prefetchers,tensorrt.prefetch_queue_size,tensorrt.inference_queue_size"
python $parser ./logs/data_prefetchers/*.log --output_params ${params}
