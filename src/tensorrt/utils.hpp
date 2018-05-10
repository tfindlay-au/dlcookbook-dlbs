/*
 (c) Copyright [2017] Hewlett Packard Enterprise Development LP
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#ifndef DLBS_TENSORRT_BACKEND_UTILS
#define DLBS_TENSORRT_BACKEND_UTILS

#include <exception>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>

void fill_random(std::vector<float>& vec);
void get_my_shard(const size_t length, const size_t num_shards, const size_t my_shard,
                  size_t& shard_pos, size_t &shard_length);
size_t get_tensor_size(const ITensor* tensor);
size_t get_binding_size(ICudaEngine* engine, const int idx);

template<typename T>
std::string S(const T& t) { return std::to_string(t); }

/**
 * https://stackoverflow.com/a/26221725/575749
 * http://www.cplusplus.com/reference/cstdio/snprintf/
 */
template<typename ... Args>
std::string fmt(const std::string& format, Args ... args) {
    size_t size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[ size ]); 
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);                   // We don't want the '\0' inside
}


// Check CUDA result.
#define cudaCheck(ans) { cudaCheckf((ans), __FILE__, __LINE__); }
inline void cudaCheckf(const cudaError_t code, const char *file, const int line, const bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/**
 * @brief Fill vector with random numbers uniformly dsitributed in [0, 1).
 * @param vec Vector to initialize.
 */
void fill_random(std::vector<float>& vec) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(std::begin(vec), std::end(vec), gen);
}

/**
 * @brief Various utility methods to work with file system, in particular, 
 * working with raw image datasets.
 */
class fs_utils {
private:
    /**
     * @brief Return human readeable 'errno' description.
     */
    static std::string get_errno() {
        return std::string(strerror(errno));
    }
public:
    /**
     */
    static std::string parent_dir(std::string dir) {
        if (dir.empty())
            return "";
        dir = normalize_path(dir);
        dir = dir.erase(dir.size() - 1);
        const auto pos = dir.find_last_of("/");
        if(pos == std::string::npos)
            return "";
        return dir.substr(0, pos);
    }
    /**
     */
    static int mk_dir(std::string dir, const mode_t mode=0700) {
        struct stat sb;
        if (stat(dir.c_str(), &sb) != 0 ) {
            // Path does not exist
            const int ret = mk_dir(parent_dir(dir), mode);
            if(ret < 0)
                return ret;
            return mkdir(dir.c_str(), mode);
        }
        if (!S_ISDIR(sb.st_mode)) {
            // Path exists and is not a directory
            return -1;
        }
        // Path exists and is a directory.
        return 0;
    }
    /**
     * @brief Makes sure that a path 'dir', which is supposed to by a directory,
     * ends with one forward slash '/'
     * @param dir A directory name.
     */
    static std::string normalize_path(std::string dir) {
        const auto pos = dir.find_last_not_of("/");
        if (pos != std::string::npos && pos + 1 < dir.size())
            dir = dir.substr(0, pos + 1);
        dir += "/";
        return dir;
    }
    /**
    * @brief Read text file \p name line by line putting lines into \p lines.
    * @param name[in] A file name.
    * @param lines[out] Vector with lines from this file.
    * @return True of file exists, false otherwise
    */
    static bool read_cache(const std::string& dir, std::vector<std::string>& fnames) {
        std::ifstream fstream(dir + "/" + "dlbs_image_cache");
        if (!fstream) return false;
        std::string fname;
        while (std::getline(fstream, fname))
            fnames.push_back(fname);
        return true;
    }
    /**
     * @brief Writes a cache with file names if that cache does not exist.
     * @param dir A dataset root directory.
     * @param fnames A list of image file names.
     * @return True if file exists or has been written, false otherwise.
     * 
     */
    static bool write_cache(const std::string& dir, std::vector<std::string>& fnames) {
        struct stat sb;
        const std::string cache_fname = dir + "/" + "dlbs_image_cache";
        if (stat(cache_fname.c_str(), &sb) == 0)
            return true;
        std::ofstream fstream(cache_fname.c_str());
        if (!fstream)
            return false;
        for (const auto& fname : fnames) {
            fstream << fname << std::endl;
        }
        return true;
    }
    /**
     * @brief Prepend \p dir to all file names in \p files
     * @param dir Full path to a dataset
     * @param files Image files with relative file paths.
     */
    static void to_absolute_paths(const std::string& dir, std::vector<std::string>& fnames) {
        for (auto& fname : fnames) {
            fname = dir + fname;
        }
    }
    /**
     * @brief Scan recursively directory \p dir and return image files. List of image files will 
     * contain paths relative to \p dir.
     * @param dir A root dataset directory.
     * @param files A list of image files.
     * @param subdir A subdirectory relative to \p dir. Used for recusrive scanning.
     * @return A list of image files found in \p dir or its subdirectories. Images files are identified
     * by relative paths from \p dir.
     */
    static void get_image_files(std::string dir, std::vector<std::string>& files, std::string subdir="") {
        // Check that directory exists
        struct stat sb;
        if (stat(dir.c_str(), &sb) != 0) {
            std::cerr << "[get_image_files       ]: skipping path ('" << dir << "') - cannot stat directory (reason: " << get_errno() << ")" << std::endl;
            return;
        }
        if (!S_ISDIR(sb.st_mode)) {
            std::cerr << "[get_image_files       ]: skipping path ('" << dir << "') - not a directory" << std::endl;
            return;
        }
        // Scan this directory for files and other directories
        const std::string abs_path = dir + subdir;
        DIR *dir_handle = opendir(abs_path.c_str());
        if (dir_handle == nullptr)
            return;
        struct dirent *de(nullptr);
        while ((de = readdir(dir_handle)) != nullptr) {
            const std::string dir_item(de->d_name);
            if (dir_item == "." || dir_item == "..") {
                continue;
            }
            bool is_file(false), is_dir(false);
            if (de->d_type != DT_UNKNOWN) {
                is_file = de->d_type == DT_REG;
                is_dir = de->d_type == DT_DIR;
            } else {
                const std::string dir_item_path = dir + subdir + dir_item;
                if (stat(dir_item_path.c_str(), &sb) != 0)
                    continue;
                is_file == S_ISREG(sb.st_mode);
                is_dir == S_ISDIR(sb.st_mode);
            }
            if (is_dir) {
                get_image_files(dir, files, subdir + dir_item + "/");
            } else if (is_file) {
                const auto pos = dir_item.find_last_of('.');
                if (pos != std::string::npos) {
                    std::string fext = dir_item.substr(pos + 1);
                    std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);
                    if (fext == "jpg" || fext == "jpeg") {
                        files.push_back(subdir + dir_item);
                    }
                }
            }
        }
        closedir(dir_handle);
    }
    
    static void initialize_dataset(std::string& data_dir, std::vector<std::string>& files) {
        data_dir = fs_utils::normalize_path(data_dir);
        if (!fs_utils::read_cache(data_dir, files)) {
            std::cout << "[image_provider        ]: found " + S(files.size()) +  " image files in file system." << std::endl;
            fs_utils::get_image_files(data_dir, files);
            if (!files.empty()) {
                if (!fs_utils::write_cache(data_dir, files)) {
                     std::cout << "[image_provider        ]: failed to write file cache." << std::endl;
                }
            }
        } else {
            std::cout << "[image_provider        ]: read " + S(files.size()) +  " from cache." << std::endl;
            if (files.empty()) { 
                std::cout << "[image_provider        ]: found empty cache file. Please, delete it and restart DLBS. " << std::endl;
            }
        }
        if (files.empty()) {
            std::cout << "[image_provider        ]: no input data found, exiting." << std::endl;
        }
        fs_utils::to_absolute_paths(data_dir, files);
    }
};


void get_my_shard(const size_t length, const size_t num_shards, const size_t my_shard,
                  size_t& shard_pos, size_t &shard_length) {
    shard_length =  length / num_shards;
    shard_pos = shard_length * my_shard;
    if (my_shard == num_shards - 1 && shard_length*num_shards != length) {
        shard_length = length - shard_length*my_shard;
    }
}

/**
 * @brief Return number of elements in \p tensor.
 * @param tensor A pointer to a tensor object.
 * @return Number of elements in \p tensor.
 */
size_t get_tensor_size(const ITensor* tensor) {
  #if NV_TENSORRT_MAJOR >= 3
  Dims shape = tensor->getDimensions();
  long sz = 1;
  for (int i=0; i<shape.nbDims; ++i) {
    sz *= shape.d[i];
  }
  return static_cast<size_t>(sz);
#else
  // Legacy TensorRT returns Dims3 object
  Dims3 shape = tensor->getDimensions();
  return long(shape.c) * shape.w * shape.h;
#endif
}

/**
 * @brief Return number of elements in tensor from binding
 * list accosiated with index \p idx.
 * @param engine Pointer to an engine.
 * @param idx Index of the tensor.
 * @return Number of elements in tensor.
 */
// Get number of elements in this tensor (blob).
size_t get_binding_size(ICudaEngine* engine, const int idx) {
#if NV_TENSORRT_MAJOR >= 3
  const Dims shape = engine->getBindingDimensions(idx);
  long sz = 1;
  for (int i=0; i<shape.nbDims; ++i) {
    sz *= shape.d[i];
  }
  return static_cast<size_t>(sz);
#else
  // Legacy TensorRT returns Dims3 object
  const Dims3 dims = engine->getBindingDimensions(idx);
  return dims.c * dims.h * dims.w;
#endif
}

/**
 * @brief Simple timer to measure execution times.
 */
class timer {
public:
  timer() {
    restart();
  }
  /**
   * @brief Restart timer.
   */
  void restart() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  /**
   * @brief Return time in milliseconds elapsed since last 
   * call to @see restart.
   */
  float ms_elapsed() const {
    const auto now = std::chrono::high_resolution_clock::now();
    return float(std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count()) / 1000.0;
  }
private:
  std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Class that can track batch processing times including pure inference
 * time and total batch time including data transfer overhead (CPU <--> GPU).
 */
class time_tracker {
private:
  // The 'inference' time is a time without CPU <--> GPU data transfers, the 'total' time
  // is a complete time including those data transfers. The 'inference' and 'total' are used
  // if report_frequency is > 0.
  // The vectors with 'all_' prefix store batch times for entire benchmark.
  std::vector<float> batch_times_;  //!< Total batch times including CPU <--> GPU data transfers.
  std::vector<float> infer_times_;  //!< Pure inference batch times excluding data transfers overhead.
  
  timer batch_timer_;  //!< Timer to measure total batch times.
  timer infer_timer_;  //!< Timer to measure pure inference times.
  
  int iter_idx_;       //!< Index when current iteration has started. Used when user requested intermidiate output.
  
  size_t num_batches_; //!< In case we need to reset the state,
public:
    /**
     * @brief Initializes time tracker.
     * @param num_batches Number of input data instances associated with
     * each element in time_tracker#batch_times_ and time_tracker#infer_times_.
     */
    explicit time_tracker(const size_t num_batches) : num_batches_(num_batches) {
        reset();
    }
    
    void reset() {
        infer_times_.reserve(num_batches_);
        batch_times_.reserve(num_batches_);
        iter_idx_ = 0;
    }
    
    void batch_started() {batch_timer_.restart();};
    void infer_started() {infer_timer_.restart();};
    void infer_done()  {infer_times_.push_back(infer_timer_.ms_elapsed());};
    void batch_done()  {batch_times_.push_back(batch_timer_.ms_elapsed());};
    /** @brief Start new iteration*/
    void new_iteration() { iter_idx_ = infer_times_.size(); }
    
    std::vector<float>& get_batch_times() { return batch_times_; }
    std::vector<float>& get_infer_times() { return infer_times_; }
    
    float last_batch_time() const { return batch_times_.back(); }
    float last_infer_time() const { return infer_times_.back(); }
    
    int get_iter_idx() const { return iter_idx_; }
};

/**
 * @brief Estimating average based on input stream without storign complete history.
 */
class running_average {
private:
    size_t i_ = 1;
    double average_ = 0;
public:
    void update(const float val) {
        average_ = average_ + (static_cast<double>(val) - average_) / i_;
        i_ ++;
    }
    double value() const { return average_; }
    size_t num_steps() const { return i_-1; }
};

/**
 * @brief The profiler, if enabled by a user, profiles execution times of 
 * individual layers.
 */
class profiler_impl : public IProfiler {
private:
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

public:
    void reset() { 
        mProfile.clear();
    }
  
    virtual void reportLayerTime(const char* layerName, float ms) {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end()) {
            mProfile.push_back(std::make_pair(layerName, ms));
        } else {
            record->second += ms;
        }
    }

    void printLayerTimes(const int num_iterations) {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++) {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / num_iterations);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / num_iterations);
    }
};

#endif
