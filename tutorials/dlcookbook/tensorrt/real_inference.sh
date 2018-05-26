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
#if false; then
python $dlbs run \
       --log-level=$loglevel\
       -Pexp.docker_args='"--rm --ipc=host --privileged"'\
       -Pruntime.launcher='"CUDA_VISIBLE_DEVICES=0,1,2,3 DLBS_TENSORRT_SYNCH_BENCHMARKS=0,2,dlbs_ipc numactl --localalloc --physcpubind=0-17"'\
       -Pexp.dtype='"float16"'\
       -Vexp.gpus='["0,1,2,3"]'\
       -Vexp.model='["resnet50"]'\
       -Vexp.replica_batch='[128]'\
       -Pexp.num_warmup_batches=100\
       -Pexp.num_batches=700\
       -Pexp.data_dir='"/dev/shm/tensorrt227/numa0/"'\
       -Ptensorrt.data_name='"tensors1"'\
       -Ptensorrt.num_prefetchers=8\
       -Ptensorrt.inference_queue_size=32\
       -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.effective_batch}_${exp.num_gpus}_0.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"' &
#fi
#if false; then
#export CUDA_VISIBLE_DEVICES=4,6
python $dlbs run \
       --log-level=$loglevel\
       -Pexp.docker_args='"--rm --ipc=host --privileged"'\
       -Pruntime.launcher='"CUDA_VISIBLE_DEVICES=4,5,6,7 DLBS_TENSORRT_SYNCH_BENCHMARKS=1,2,dlbs_ipc  numactl --localalloc --physcpubind=18-35 "'\
       -Pexp.dtype='"float16"'\
       -Vexp.gpus='["0,1,2,3"]'\
       -Vexp.model='["resnet50"]'\
       -Vexp.replica_batch='[128]'\
       -Pexp.num_warmup_batches=100\
       -Pexp.num_batches=700\
       -Pexp.data_dir='"/dev/shm/tensorrt227/numa1"'\
       -Ptensorrt.data_name='"tensors1"'\
       -Ptensorrt.num_prefetchers=8\
       -Ptensorrt.inference_queue_size=32\
       -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.effective_batch}_${exp.num_gpus}_1.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"' &
#fi
wait
params="exp.status,exp.framework_title,exp.replica_batch,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus,exp.dtype,results.mgpu_effective_throughput"
python $parser ./logs/real/*.log --output_params ${params}

#-Pexp.data_dir='"/lvol/serebrya/datasets/tensorrt_uchar_chunks"'\

