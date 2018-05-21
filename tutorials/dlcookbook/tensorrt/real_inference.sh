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
       -Pruntime.launcher='"TENSORRT_USE_PINNED_MEMORY=1 "'\
       -Pexp.dtype='"float16"'\
       -Vexp.gpus='["0"]'\
       -Vexp.model='["resnet50"]'\
       -Pexp.replica_batch=256\
       -Pexp.num_warmup_batches=20\
       -Pexp.num_batches=250\
       -Pexp.data_dir='"/dev/shm/tensorrt_uchar_chunks/1"'\
       -Ptensorrt.data_name='"tensors1"'\
       -Ptensorrt.num_prefetchers=3\
       -Ptensorrt.inference_queue_size=6\
       -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.num_gpus}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus"
python $parser ./logs/real/*.log --output_params ${params}

#-Pexp.data_dir='"/lvol/serebrya/datasets/tensorrt_uchar_chunks"'\

