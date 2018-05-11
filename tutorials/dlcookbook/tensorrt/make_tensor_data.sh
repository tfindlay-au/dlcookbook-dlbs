#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
# Run multi-GPU inference with real data.
#------------------------------------------------------------------------------#


docker run --rm \
           -ti \
	   --volume=/lvol/serebrya/datasets/train:/lvol/serebrya/datasets/train  \
	   --volume=/lvol/serebrya/datasets/tensorrt:/lvol/serebrya/datasets/tensorrt \
           hpe/tensorrt:cuda9-cudnn7 \
           tensorrt
exit 0

