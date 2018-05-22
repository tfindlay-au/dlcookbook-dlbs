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
rm -rf ./logs/synthetic_mprocess
mkdir -p ./logs/synthetic_mprocess

gpus="0,4";
cores="0-17,18-35"
ranks="0,1"
cores=(${cores//,/ });
gpus=(${gpus//,/ });
ranks=(${ranks//,/ })
for i in "${!gpus[@]}"
do
    gpu=${gpus[$i]}
    core=${cores[$i]}
    rank=${ranks[$i]}

    #taskset -c $core \
    numactl --localalloc --physcpubind=$core \
    python $dlbs run \
           --log-level=$loglevel\
           -Pruntime.launcher='"TENSORRT_SYNCH_BENCHMARKS=${tensorrt.rank},2,dlbs_ipc "'\
           -Pexp.docker_args='"--rm --ipc=host"'\
           -Pexp.dtype='"float16"'\
           -Pexp.gpus=\"$gpu\"\
           -Vexp.model='["resnet50"]'\
           -Pexp.replica_batch=128\
           -Pexp.num_warmup_batches=50\
           -Pexp.num_batches=300\
           -Ptensorrt.inference_queue_size=4\
           -Ptensorrt.rank=$rank\
           -Pexp.log_file='"${BENCH_ROOT}/logs/synthetic_mprocess/${exp.model}_${exp.gpus}.log"'\
           -Pexp.phase='"inference"'\
           -Pexp.docker=true\
           -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
           -Pexp.framework='"tensorrt"' &
done
wait

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus,results.mgpu_effective_throughput"
python $parser ./logs/synthetic_mprocess/*.log --output_params ${params}
