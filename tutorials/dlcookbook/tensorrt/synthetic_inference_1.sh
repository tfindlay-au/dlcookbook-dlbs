#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../../scripts/environment.sh
dlbs=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
loglevel=warning
#------------------------------------------------------------------------------#
# Run multi-GPU inference with synthetic data.
#------------------------------------------------------------------------------#
rm -rf ./logs/synthetic

python $dlbs run \
       --log-level=$loglevel\
       -Ptensorrt.data_dir='""'\
       -Pexp.dtype='"float16"'\
       -Pruntime.launcher='"TENSORRT_INFERENCE_IMPL_VER=latest"'\
       -Vexp.gpus='["0"]'\
       -Vexp.model='["alexnet_owt"]'\
       -Vexp.replica_batch='[32]'\
       -Pexp.num_warmup_batches=10\
       -Pexp.num_batches=10\
       -Ptensorrt.inference_queue_size=8\
       -Pexp.log_file='"${BENCH_ROOT}/logs/synthetic/${exp.model}_${exp.dtype}_${exp.replica_batch}_${exp.num_gpus}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.dtype,exp.gpus"
python $parser ./logs/synthetic/*.log --output_params ${params}
