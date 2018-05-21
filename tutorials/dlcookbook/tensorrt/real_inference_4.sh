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
# Change #GPUs in 4 places!
rm -rf ./logs/real
mkdir -p ./logs/real

gpus="0,2,4,6";
cores="0-8,9-17,18-26,27-35"
cores=(${cores//,/ });
gpus=(${gpus//,/ });
for i in "${!gpus[@]}"
do
    gpu=${gpus[$i]}
    core=${cores[$i]}

    #taskset -c $core \
    numactl --localalloc --physcpubind=$core \
    python $dlbs run \
           --log-level=$loglevel\
           -Pruntime.launcher='"TENSORRT_USE_PINNED_MEMORY=1 TENSORRT_DO_NOT_OVERLAP_COPY_COMPUTE=0 TENSORRT_INFERENCE_IMPL_VER=0"'\
           -Ptensorrt.num_prefetchers=5\
           -Pexp.data_dir='"/dev/shm/tensorrt_uchar_chunks/${exp.gpus}"'\
           -Ptensorrt.data_name='"tensors1"'\
           -Pexp.dtype='"float16"'\
           -Pexp.gpus=\"$gpu\"\
           -Vexp.model='["resnet50"]'\
           -Pexp.replica_batch=256\
           -Pexp.num_warmup_batches=50\
           -Pexp.num_batches=250\
           -Ptensorrt.inference_queue_size=4\
           -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.gpus}.log"'\
           -Pexp.phase='"inference"'\
           -Pexp.docker=true\
           -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
           -Pexp.framework='"tensorrt"' &
done
wait

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus"
python $parser ./logs/real/*.log --output_params ${params}
