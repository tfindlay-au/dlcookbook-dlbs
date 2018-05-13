
#include "core/dataset/tensor_dataset.hpp"

/**
 --------------------------------------
 #prefetchers   throughput (images/sec)
      1                19570.7
      2                43529.5
      3                65412.9
      4                81726.3
      5                93494.4
      6                101093
---------------------------------------

 */
int main(int argc, char **argv) {
    std::vector<int> prefetchers;
    std::vector<float> throughput;
    
    for (int num_prefetchers=1; num_prefetchers<=6; num_prefetchers++) {
        const auto images_sec = tensor_dataset::benchmark(
            "/dev/shm/fast",         // Dataset path
            512,                     // Batch size
            227,                     // Image size
            num_prefetchers,         // Number of batches to prefetch in parallel
            num_prefetchers*3,       // Number of preallocated batches
            50,                      // Warmup batches
            300                      // Benchmark batches
        );
        prefetchers.push_back(num_prefetchers);
        throughput.push_back(images_sec);
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    for (size_t i=0; i<prefetchers.size(); ++i) {
        std::cout << prefetchers[i] << "\t" << throughput[i] << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
}