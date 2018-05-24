#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Convert raw images into a tensor representation.
#------------------------------------------------------------------------------#
data_dir=/lvol/serebrya/datasets/tensorrt227/uchar
dtype=uchar
img_size=227
batch_size=512
num_prefetchers=8
num_preallocated_batches=36
num_warmup_batches=500
num_batches=4000

# We will run this command in a container. Do not change this line.
exec="benchmark_tensor_dataset --data_dir /mnt/dataset --batch_size=${batch_size} --img_size ${img_size} --num_prefetchers=${num_prefetchers} --dtype=${dtype}"
exec="$exec --prefetch_pool_size=${num_preallocated_batches} --num_warmup_batches=${num_warmup_batches} --num_batches=${num_batches}"

docker run  -ti \
            --rm \
            --volume=${data_dir}:/mnt/dataset  \
            hpe/tensorrt:cuda9-cudnn7 \
           /bin/bash -c "${exec}"
exit 0
