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
# Change #GPUs in 4 places!
rm -rf ./logs/real_mprocess/8
mkdir -p ./logs/real_mprocess/8

gpus="0,1,2,3,4,5,6,7";
gpus=(${gpus//,/ });
for gpu in "${gpus[@]}"
do
    python $dlbs run \
           --log-level=$loglevel\
           -Pexp.dtype='"float32"'\
           -Pexp.gpus=\"$gpu\"\
           -Vexp.model='["alexnet_owt"]'\
           -Pexp.replica_batch=512\
           -Pexp.num_warmup_batches=100\
           -Pexp.num_batches=400\
           -Pexp.data_dir='"/lvol/serebrya/datasets/tensorrt"'\
           -Ptensorrt.data_name='"tensors"'\
           -Ptensorrt.num_prefetchers=3\
           -Ptensorrt.num_decoders=2\
           -Ptensorrt.prefetch_queue_size=32\
           -Ptensorrt.prefetch_batch_size=32\
           -Ptensorrt.inference_queue_size=6\
           -Pexp.log_file='"${BENCH_ROOT}/logs/real_mprocess/8/${exp.model}_${exp.gpus}.log"'\
           -Pexp.phase='"inference"'\
           -Pexp.docker=true\
           -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
           -Pexp.framework='"tensorrt"' &
done
wait

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
python $parser ./logs/real_mprocess/8/*.log --output_params ${params}
