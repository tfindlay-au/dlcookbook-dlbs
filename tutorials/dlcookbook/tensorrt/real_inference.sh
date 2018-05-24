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
mkdir -p ./logs/real
#numactl --localalloc --physcpubind=0-17 \
python $dlbs run \
       --log-level=$loglevel\
       -Pexp.docker_args='"--rm --ipc=host"'\
       -Pexp.dtype='"float32"'\
       -Vexp.gpus='["0,1,2,3,4,5,6,7"]'\
       -Vexp.model='["resnet50"]'\
       -Vexp.replica_batch='[256,128]'\
       -Pexp.num_warmup_batches=100\
       -Pexp.num_batches=350\
       -Pexp.data_dir='"/lvol/serebrya/datasets/tensorrt227/uchar"'\
       -Ptensorrt.data_name='"tensors1"'\
       -Ptensorrt.num_prefetchers=8\
       -Ptensorrt.inference_queue_size=32\
       -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.effective_batch}_${exp.num_gpus}_0.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

if false; then
numactl --localalloc --physcpubind=18-35 \
python $dlbs run \
       --log-level=$loglevel\
       -Pruntime.launcher='"TENSORRT_SYNCH_BENCHMARKS=1,2,dlbs_ipc  "'\
       -Pexp.docker_args='"--rm --ipc=host"'\
       -Pexp.dtype='"float16"'\
       -Vexp.gpus='["4"]'\
       -Vexp.model='["alexnet_owt"]'\
       -Vexp.replica_batch='[1024]'\
       -Pexp.num_warmup_batches=50\
       -Pexp.num_batches=500\
       -Pexp.data_dir='"/dev/shm/tensorrt227/numa1"'\
       -Ptensorrt.data_name='"tensors1"'\
       -Ptensorrt.num_prefetchers=8\
       -Ptensorrt.inference_queue_size=16\
       -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.effective_batch}_${exp.num_gpus}_1.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"' &
fi
#wait
params="exp.status,exp.framework_title,exp.replica_batch,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus,exp.dtype"
python $parser ./logs/real/*.log --output_params ${params}

#-Pexp.data_dir='"/lvol/serebrya/datasets/tensorrt_uchar_chunks"'\

