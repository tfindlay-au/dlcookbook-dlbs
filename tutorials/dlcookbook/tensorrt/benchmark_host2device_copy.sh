#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Benchmark host to device memory copy. This will provide an intuition on what
# maximal throughput can be achieved taking into account only ability of a system
# to stream data from host to device memory.
# For instance, let's consider that this benchmark says that you can stream data
# at 10.5 GB/sec (not very good taking into account 16 PCIe-3 lanes, but anyway).
# Assuming that data is transfered as floating point arrays of length
# BatchSize*3*Wight*Height where Wight=Height=227 (AlexNet), then you can expect
# inference throughput at most 10.5*1024^3/(3*227^2*4)~18000 images/sec. This,
# however, will never be achieved due to various factors: (1) compute intensive
# neural network model which cannot be processed at this rate and (2) software
# implementation of an inference function that does not optimally overlap all
# comptue/copy operations.
#------------------------------------------------------------------------------#
gpu=0                  # GPU index to use.
size=10                # Size of a data chunk in MegaBytes. During inference benchmarks,
                       # data is transfered as arrays of shape [BatchSize, 3, Wight, Height]
                       # of 'float' data type. These are typical sizes for AlexNetOWT where
                       # Width = Height = 227:
                       #  Batch size (images):  32 	64 	128	 256	 512	 1024
                       #  Batch size (MB):      19	 38	  75	 151	 302	  604
pinned_mem=true        # Allocate buffer in host pinned memory.
                       # pageable memory.
num_warmup_batches=10  # Number of warmup iterations.
num_batches=100        # Number of benchmark iterations.



[ "$pinned_mem" == "true" ] && pinned="--pinned" || pinned=""
# We will run this command in a container. Do not change this line.
exec="benchmark_host2device_copy --gpu=$gpu --size=$size $pinned --num_warmup_batches=${num_warmup_batches} --num_batches=${num_batches}"
docker run -ti --rm hpe/tensorrt:cuda9-cudnn7 /bin/bash -c "${exec}"
exit 0
