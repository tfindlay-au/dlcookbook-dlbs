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
rm -rf ./logs/real
python $dlbs run \
       --log-level=$loglevel\
       -Vexp.gpus='["0"]'\
       -Vexp.model='["alexnet_owt"]'\
       -Pexp.replica_batch=128\
       -Pexp.num_warmup_batches=10\
       -Pexp.num_batches=20\
       -Pexp.data_dir='"/home/serebrya/data/fast"'\
       -Ptensorrt.data_name='"tensors"'\
       -Ptensorrt.num_prefetchers=1\
       -Ptensorrt.num_decoders=2\
       -Ptensorrt.prefetch_queue_size=32\
       -Ptensorrt.prefetch_batch_size=32\
       -Ptensorrt.inference_queue_size=10\
       -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.num_gpus}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
python $parser ./logs/real/*.log --output_params ${params}
