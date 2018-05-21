# DLBS TensorRT backend tools

The TensorRT backend provides several tools to benchmark inference phase with
synthetic/real data on single/multi GPU machine.

## Synthetic data
Synthetic data is a random data in host memory. Benchmarks with synthetic data
do not include input data pipeline and overhead associated with it such as
storage/network constraints, image preprocessing overhead, memory bandwidth
limitations etc. Benchmarks of this kind demonstrate the upper bound of a achievable
performance. Since synthetic input data is stored in host memory, the benchmarking
tool copies data to/from GPU memory and this is taken into account. The benchmarking
tool reports both `inference` time (when data in GPU memory) and `batch` time that
includes CPU<-->GPU traffic overhead.

## Real data
The TensorRT backend supports real data to benchmark storage, network, memory etc
components that are involved in inference phase. Our primary use case here is a
validation inference on very large datasets. Something, that's typical for, for
instance, self driving cars companies that need to validate their models continuously.

The backend supports two types of datasets - images (JPEGs) and preprocessed images
stored as float/unsigned char tensors in large files. Depending on a dataset, a
bottlneck component could be either storage or CPU.

### Image Dataset
This is a collection of JPEG files similar to ImageNet.Every image is stored in its
own file. Images are loaded and processed by OpenCV library. Default behavior is to
crop images to appropriate size, and if it's not possible, resize. Depending on where
exactly such dataset is stored, a bottleneck could be a storage (reading many small
files), or CPU (resizing images). This can be critical for not very compute intensive
models such as AlexNetOWT or for systems with 8 or more GPUs.

### Tensor Dataset
This type of dataset stores images as tensors in large binary files. The goal of it
is to put stress on storage/network infrastructure and completely unload compute from
CPUs. The backend provides two tools to work with this dataset - a coverter
(`images2tensors`) that converts images and `benchmark_tensor_dataset` that can be
used to stream images to host memory without doing inference.

Since the primary goal here is to determine storage limits, the dataset format is very
simple and cannot be used as is in any real world applications.

#### Images2Tensors tool
Converts images (JPEGs) to a binary representation that can directly be used
by the inference engine.

The tool is configured with the following comamnd line arguments:
1. `--input_dir` Input directory. This directory must exist and must contain
images (jpg, jpeg) in that directory or one of its sub-directories. ImageNet
directory with raw images is one example of a valid directory structure.
2. `--output_dir` Output directory. The tool will write output files in this
directory depending on input parameter `images_per_file` that determines how many
images each output file must contain. If it's 1, then this directory will have
exactly the same structure as input directory. Each input file will get same
relative path, will have same name and extension. Even though file extension will
remain the same, the content will be different. It will not be a valid image files
but a C array of float or unsigned char numbers of shape [3, Size, Size]. If user
wants to use large files with many images in each file, the output directory will
have flat structure with large binary files with extension `*.tensors`. Each file
will contain N images of shape [3, Size, Size]. No label or any other info will be
stored in output files.
3. `--size` Resize images to this size. Output images will have square shape
[3, size, size].
4. `--dtype` A data type for a matrix storage. Two types are supported: 'float'
and 'uchar'. The 'float' is a single precision 4 byte storage. Images take more
space but are read directly into an inference buffer. The 'uchar' (unsigned char)
is a one byte storage that takes less disk space but needs to be converted from
unsigned char to float array.
5. `--shuffle` Shuffle list if images. Useful with combination `--nimages` to
convert only a small random subset.
6. `--nimages` If nimages > 0, only convert this number of images. Use `--shuffle`
to randomly shuffle list of images with this option.
7. `--nthreads` Use this number of threads to convert images. This will significantly
increase overall throughput.
8. `--images_per_file` Number of images per file. If this value is 1, images2tensors
will create the same directory structure with the same file names in `--input_dir`.
If this value is greater than 1, images2tensors will create a flat directory with
\*.tensors files.

#### Benchmark tensor dataset
This tool can be used to benchmark storage/network by simply streaming images
from where they are stored to host memory. The following command line arguments
are supported:
1. `--data_dir` Path to a dataset to use.
2. `--batch_size` Create batches of this size.
3. `--img_size` Size of images in a dataset (width = height).
4. `--num_prefetchers` Number of prefetchers (data readers).
5. `--prefetch_pool_size` Number of pre-allocated batches. Memory for batches is
preallocated in advance and then is resused by prefetchers.
6. `--num_warmup_batches` Number of warmup iterations.
7. `--num_batches` Number of benchmark iterations.
8. `--dtype`Tensor data type - 'float' or 'uchar'.
