#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Benchmark host to device memory copy.
#------------------------------------------------------------------------------#
gpu=0                  # GPU index to use.
size=10                # Size of a data chunk in MegaBytes.
pinned="--pinned"      # Allocate buffer in host pinned memory. Set it to empty value to use
                       # pageable memory.
num_warmup_batches=10  # Number of warmup iterations.
num_batches=100        # Number of benchmark iterations.


# We will run this command in a container. Do not change this line.
exec="benchmark_host2device_copy --gpu=$gpu --size=$size $pinned --num_warmup_batches=${num_warmup_batches} --num_batches=${num_batches}"
docker run  -ti --rm hpe/tensorrt:cuda9-cudnn7 /bin/bash -c "${exec}"
exit 0
