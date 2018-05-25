#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Convert raw images into a binary representation.
#------------------------------------------------------------------------------#
input_dir=/path/to/jpeg/images  # Path to an input data. Must exist and contain JPEG files.
output_dir=/output/folder       # Output folder.
dtype=uchar                     # Data type for image arrays in output files:
                                #     'float' - 4 bytes
                                #     'uchar' (unsigned char) - 1 byte
img_size=227                    # Size of output images. Now, it must be the same
                                # size as accepted by a neural network model. With
                                # AlexNet, it's 227.

images_per_file=1               # Number of images per one file.

nimages=0                       # Set this value to > 0 to convert only this number
                                # of images.
shuffle=""                      # Default option is to not shuffle list of input images.
                                # This option is useless without nimages option > 0.
nthreads=5    # Number of workers. You may want to increase this number if your
              # dataset is large. Each worker will operate on its own unique set of
              # input files. No two workers write the same output file. Select number
              # of threads and images per file depending on size of your dataset.

# We will run this command in a container. Do not change this line.
exec="images2tensors --input_dir /mnt/input --output_dir /mnt/output --size ${img_size}"
exec="${exec} --nimages ${nimages} ${shuffle} --nthreads ${nthreads}"
exex="${exec} --images_per_file ${images_per_file} --dtype=${dtype}"

mkdir -p ${output_dir}
docker run  -ti \
            --rm \
            --volume=${input_dir}:/mnt/input  \
            --volume=${output_dir}:/mnt/output \
            hpe/tensorrt:cuda9-cudnn7 \
            /bin/bash -c "${exec}"
exit 0
