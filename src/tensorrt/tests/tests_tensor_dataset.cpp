
#include "core/dataset/tensor_dataset.hpp"

/**
My development box
    /dev/sda (Hitachi HDS72105)
        2018-05-14 06:58:26           INFO [benchmarks            ]: num_readers=1, throughput=28.92
        2018-05-14 07:00:27           INFO [benchmarks            ]: num_readers=2, throughput=542.49
        2018-05-14 07:01:51           INFO [benchmarks            ]: num_readers=3, throughput=729.96
        2018-05-14 07:03:51           INFO [benchmarks            ]: num_readers=4, throughput=555.15
        2018-05-14 07:07:06           INFO [benchmarks            ]: num_readers=5, throughput=318.20
        2018-05-14 07:12:52           INFO [benchmarks            ]: num_readers=6, throughput=159.57
        2018-05-14 07:20:22           INFO [benchmarks            ]: num_readers=7, throughput=143.80
        2018-05-14 07:30:47           INFO [benchmarks            ]: num_readers=8, throughput=86.39
        2018-05-14 07:50:16           INFO [benchmarks            ]: num_readers=9, throughput=47.47
    /dev/sda (Hitachi HDS72105) 1 thread, 2 batches per file
        |--------------------------------------------------|
        |             |              throughput            |
        | num_readers |    images/sec)    |     MB/sec     |
        |--------------------------------------------------|
        |     1       |     188           |    111.1021    |    1 thread, 2 batches per file
        |--------------------------------------------------|

    /dev/shm (1 image per file)
        |--------------------------------------------------|
        |             |              throughput            |
        | num_readers |    images/sec)    |     MB/sec     |
        |--------------------------------------------------|
        |     1       |     18440         |    10874.4834  |
        |     2       |     41209         |    24301.2129  |
        |     3       |     61095         |    36028.4453  |
        |     4       |     77747         |    45847.9766  |
        |     5       |     88551         |    52219.0117  |
        |     6       |     97757         |    57648.0742  |
        |     7       |     103662        |    61129.9883  |
        |     8       |     101845        |    60058.4688  |
        |     9       |     101370        |    59778.2031  |
        |     10      |     99186         |    58490.3438  |
        |--------------------------------------------------|
 */
int main(int argc, char **argv) {
    int batch_sz = 512,
        img_sz = 227,
        num_prefetchers_to_use(2);
    std::string data_dir = "/home/serebrya/data/tensors-float-8196";
    
    
    std::vector<int> prefetchers;
    std::vector<float> throughput;    
    for (int num_prefetchers=num_prefetchers_to_use; num_prefetchers<=num_prefetchers_to_use; num_prefetchers++) {
        const auto images_sec = tensor_dataset::benchmark(
            data_dir,         // Dataset path
            batch_sz,                // Batch size
            img_sz,                  // Image size
            num_prefetchers,         // Number of batches to prefetch in parallel
            num_prefetchers*3,       // Number of preallocated batches
            10,                      // Warmup batches
            50                      // Benchmark batches
        );
        prefetchers.push_back(num_prefetchers);
        throughput.push_back(images_sec);
    }
    const float img_mb = float(3*img_sz*img_sz)*4 / (1024*1024);
    std::cout << "|--------------------------------------------------|" << std::endl;
    std::cout << "|             |              throughput            |" << std::endl;
    std::cout << "| num_readers |    images/sec)    |     MB/sec     |" << std::endl;
    std::cout << "|--------------------------------------------------|" << std::endl;
    for (size_t i=0; i<prefetchers.size(); ++i) {
        const float achieved_throughput = throughput[i]*img_mb;
        printf("|     %-8d|     %-14d|    %-12.4f|\n", prefetchers[i], int(throughput[i]), achieved_throughput);
    }
    std::cout << "|--------------------------------------------------|" << std::endl;
}