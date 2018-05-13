#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Convert raw images into a tensor representation.
#------------------------------------------------------------------------------#
input_dir=/path/to/input/dataset   # Path to an input data.
output_dir=/path/to/outut/folder   # Output folder.
img_size=227                       # Size of output images. Now, it must be the same
                                   # size as accepted by a neural network model. With
                                   # AlexNet, it's 227.

nimages=0                          # Set this value to > 0 to convert only this number
                                   # of images.
shuffle=""                         # Default option is to not shuffle list of input images.
                                   # This option is useless without nimages option > 0.
#shuffle="--shuffle"               # If nimages > 0, you may want to randomly shuffle
                                   # list of input images.


nthreads=4    # Number of worker. You may want to increase this number if your
              # dataset is large. On my dev box I am getting ~300-400 images/sec
              # with one worker.

# We will run this command in a container. Do not change this line.
exec="images2tensors --input_dir /mnt/input --output_dir /mnt/output --size ${img_size} --nimages ${nimages} ${shuffle} --nthreads ${nthreads}"

mkdir -p ${output_dir}
docker run  -ti \
           --rm \
	          --volume=${input_dir}:/mnt/input  \
	          --volume=${output_dir}:/mnt/output \
           hpe/tensorrt:cuda9-cudnn7 \
           /bin/bash -c "${exec}"
exit 0
