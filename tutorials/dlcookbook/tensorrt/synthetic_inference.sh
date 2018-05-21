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
       -Pexp.dtype='"int8"'\
       -Pruntime.launcher='"TENSORRT_USE_PINNED_MEMORY=1 TENSORRT_INFERENCE_IMPL_VER=0"'\
       -Vexp.gpus='["0"]'\
       -Vexp.model='["resnet50"]'\
       -Pexp.replica_batch=128\
       -Pexp.num_warmup_batches=20\
       -Pexp.num_batches=250\
       -Ptensorrt.inference_queue_size=8\
       -Pexp.log_file='"${BENCH_ROOT}/logs/synthetic/${exp.model}_${exp.num_gpus}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
python $parser ./logs/synthetic/*.log --output_params ${params}
