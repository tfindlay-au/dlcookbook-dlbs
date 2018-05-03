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

#ifndef DLBS_TENSORRT_BACKEND_CALIBRATOR
#define DLBS_TENSORRT_BACKEND_CALIBRATOR

#include "utils.hpp"

/**
 * TODO: Is it true that we can run first all networks with small batch sizes to create fake calibration 
 *       caches and then just load that. The problem is that with large batch size calibration memory takes
 *       too much memory, for instance, an input image of 3x255x255x512 = 381 MB where 512 is a batch size.
 */
#if NV_TENSORRT_MAJOR >= 3
class tensorrt_calibrator : public IInt8LegacyCalibrator {
#else
class tensorrt_calibrator : public IInt8Calibrator {
#endif
public:
  // The batch size is for a calibration stage.
  int getBatchSize() const override { 
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::getBatchSize() { return " << batch_size_ << "; }";
    log(stream.str());
    return batch_size_; 
  }
  // For ImageNet networks and MNIST, 500 images is a reasonable size for the calibration set.
  // In a simpliest case, nbBindings is 1 and names[0] = 'data'
  // For each input tensor, a pointer to input data in GPU memory must be written into the bindings 
  // array. The names array contains the names of the input tensors, and the position for each tensor 
  // in the bindings array matches the position of its name in the names array. Both arrays have size 
  // nbBindings.
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override { 
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::getBatch(names=[";
    for (int i=0; i<nbBindings; ++i) {
      if (i != 0) {
        stream << ", ";
      }
      stream << "'" << names[i]  << "'";
    }
    stream << "], nbBindings=" << nbBindings << ")";
    log(stream.str());
    if (nbBindings != 1) {
      std::cout << "***ERROR*** Exactly one input must present but found " << nbBindings << " input(s)." << std::endl;
      exit(1);
    }
    // Make sure that this calibrator was initialized.
    if (num_batches_ <= 0) {
      std::cout << "***ERROR***: Suspicious number of batches (0) in calibrator::getBatch()." << std::endl;
      exit(1);
    }
    // Lazy memory allocation - allocate only if we are here.
    if (gpu_batch_ == nullptr) {
      host_batch_.resize(batch_size_ * input_size_);
      cudaCheck(cudaMalloc(&(gpu_batch_), sizeof(float) * batch_size_ * input_size_));
    }
    //
    if (next_batch_ >= num_batches_) { return false; }
    //
    fill_random(host_batch_);
    cudaCheck(cudaMemcpy(gpu_batch_, host_batch_.data(), sizeof(float) * host_batch_.size(), cudaMemcpyHostToDevice));
    bindings[0] = gpu_batch_;
    next_batch_ ++;
    return true; 
  }
  // The cutoff and quantile parameters take values in the range [0,1]; their meaning is discussed 
  // in detail in the accompanying white paper. To find the best calibration parameters, it can be 
  // useful to search over the parameter combinations and score the network for each combination 
  // using some additional images. searchCalibrations() illustrates how to do this. For ImageNet 
  // networks, 5000 images were used to find the optimal calibration. Since the calibration process 
  // will run many times varying only the regression and cutoff parameters, histogram caching is 
  // strongly recommended.
  double getQuantile() const override { 
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::getQuantile() { return " << quantile_ << "; }";
    log(stream.str());
    return quantile_; 
  }
  double getRegressionCutoff() const override { 
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::getRegressionCutoff() { return " << cutoff_ << "; }";
    log(stream.str());
    return cutoff_; 
  }
  const void* readCalibrationCache(size_t& length/*output param*/) override {
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::readCalibrationCache()";
    log(stream.str());
    update_cache(calibration_cache_, calibration_cache_length_, get_cache_file("calibration"));
    length = calibration_cache_length_;
    if (calibration_cache_ != nullptr) {
      std::cout << "Calibration cache has succesfully been read (length=" << length << ")." << std::endl;
    }
    return (const void*)calibration_cache_;
  }
  void writeCalibrationCache(const void* ptr, size_t length) override {
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::writeCalibrationCache(length=" << length << ")";
    log(stream.str());
    write_data(get_cache_file("calibration"), ptr, length);
  }
  const void* readHistogramCache(size_t& length/*output param*/) override { 
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::readHistogramCache()";
    log(stream.str());
    update_cache(histogram_cache_, histogram_cache_length_, get_cache_file("histogram"));
    length = histogram_cache_length_;
    if (histogram_cache_ != nullptr) {
      std::cout << "Histogram cache has succesfully been read(length=" << length << ")." << std::endl;
    }
    return (const void*)histogram_cache_;
  }
  void writeHistogramCache(const void* ptr, size_t length) override { 
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::writeHistogramCache(length=" << length << ")";
    log(stream.str());
    write_data(get_cache_file("histogram"), ptr, length);
  }
  void setLog(const bool do_log=true) {
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::setLog(do_log=" << do_log << ")";
    log(stream.str());
    do_log_ = do_log; 
  }
  void setBatchSize(const int batch_size) {
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::setBatchSize(batch_size=" << batch_size << ")";
    log(stream.str());
    batch_size_ = batch_size;
  }
  /**
   * Initialize Calibrator. No memory allocations are done here.
   * @param input_shape The shape of single input instance. Does not include batch dimension.
   * @param num_batches Numebr of calibration iterations.
   * @param model A neural network model identifier such as alexnet, resnet101, vgg13 etc.
   * @param calibration_cache_path A path to folder that contains calibration cache data. With every model two 
   * files are associated - calibration and hostogram cache files.
   */
  void initialize(const long input_size, const int num_batches, const std::string& model, const std::string& cache_path) {
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::initialize(input_size=" << input_size
           << ", num_batches=" << num_batches << ", model = " << model << ")";
    log(stream.str());

    input_size_ = input_size;
    num_batches_ = num_batches;
    next_batch_ = 0;

    cache_path_ = cache_path;
    model_ = model;

    if (cache_path_ == "") {
      std::cout << "***WARNING***: Calibration cache path is not set." << std::endl;
    } else {
      std::cout << "Calibration cache file: " << get_cache_file("calibration") << std::endl;
      std::cout << "Histogram cache file: " << get_cache_file("histogram") << std::endl;
    }
  }
  void freeCalibrationMemory() {
    log("[calibrator] Calibrator::freeCalibrationMemory");
    cudaFree(gpu_batch_);
    host_batch_.clear();
    if (calibration_cache_ != nullptr) {
      delete [] calibration_cache_;
      calibration_cache_length_ = 0;
    }
    if (histogram_cache_ != nullptr) {
      delete [] histogram_cache_;
      histogram_cache_length_ = 0;
    }
  }
private:
  void log(const std::string& msg) const {
    if (do_log_) {
      std::cout << msg << std::endl;
    }
  }
  std::string get_cache_file(const std::string& suffix) const {
    if (cache_path_ != "" && model_ != "") {
      return cache_path_ + "/" + model_ + "_" + suffix + ".bin";
    }
    return "";
  }
  void write_data(const std::string& fname, const void* ptr, size_t length) {
    if (fname != "") {
      std::ofstream file(fname.c_str(), std::ios::binary);
      if (file.is_open()) {
        file.write((char*)ptr, length);
      }
    }
  }
  char* read_data(const std::string& fname, size_t& data_length) {
    if (fname != "") {
      std::ifstream file(fname.c_str(), std::ios::binary|std::ios::ate);
      if (file.is_open()) {
        data_length = file.tellg();
        char* data = new char[data_length];
        file.seekg(0, std::ios::beg);
        file.read(data, data_length);
        return data;
      }
    }
    data_length = 0;
    return nullptr;
  }
  void update_cache(char*& cache_data, size_t& cache_length, const std::string& fname) {
    if (cache_data == nullptr) {
      cache_data = read_data(fname, cache_length);
    }
  }
private:
  int batch_size_ = 0;                  // Batch size (number of instances)
  long input_size_ = 0;                 // Size of one instance (multiplication of all dimensions)
  int num_batches_ = 0;                 // Number of batches to use for calibration
  int next_batch_ = 0;                  // During calibration, index of the next batch
  std::vector<float> host_batch_;       // Batch data in host memory
  void *gpu_batch_ = nullptr;           // Batch data in GPU memory
  
  double quantile_ = 0.5;
  double cutoff_ = 0.5;
  
  bool do_log_ = true;
  
  std::string cache_path_;  // Path to calibration cache.
  std::string model_;                   // Neural network model (to save/load calibration caches).
  char* calibration_cache_ = nullptr;   // Calibration cache loaded from file.
  size_t calibration_cache_length_ = 0;
  char* histogram_cache_ = nullptr;     // Histogram cache loaded from file.
  size_t histogram_cache_length_ = 0;
};

#endif