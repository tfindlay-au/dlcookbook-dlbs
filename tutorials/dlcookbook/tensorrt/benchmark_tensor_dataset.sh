#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Warmup system and/or run storage/network benchmarks by streaming images into
# host memory and measuring the throughput.
#------------------------------------------------------------------------------#
data_dir=/path/to/input/dataset   # This datset needs to be created with images2tensors tool.
                                  # Use ./make_tensor_dataset.sh script to create this data.
dtype=uchar                       # Data type for image arrays in dataset:
                                  #     'float' - 4 bytes
                                  #     'uchar' (unsigned char) - 1 byte
img_size=227                      # Size of input images [3, img_size, img_size] in dataset.
batch_size=512                    # Read data in batches using this batch size.
num_prefetchers=2                 # Number of parallel threads reading data from dataset.
num_preallocated_batches=4        # Number of preallocated batches. Readers do not allocate memory,
                                  # rather, they use preallocated memory from pool of batches.
num_warmup_batches=500            # Number of warmup iterations.
num_batches=4000                  # Number of benchmark iterations.

# We will run this command in a container. Do not change this line.
exec="benchmark_tensor_dataset --data_dir /mnt/dataset --batch_size=${batch_size}"
exec="${exec} --img_size ${img_size} --num_prefetchers=${num_prefetchers} --dtype=${dtype}"
exec="${exec} --prefetch_pool_size=${num_preallocated_batches} --num_warmup_batches=${num_warmup_batches}"
exec="${exec} --num_batches=${num_batches}"

docker run  -ti \
            --rm \
            --volume=${data_dir}:/mnt/dataset  \
            hpe/tensorrt:cuda9-cudnn7 \
            /bin/bash -c "${exec}"
exit 0
