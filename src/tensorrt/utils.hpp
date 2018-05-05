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
 * @brief Return number of elements in \p tensor.
 * @param tensor A pointer to a tensor object.
 * @return Number of elements in \p tensor.
 */
long get_tensor_size(const ITensor* tensor) {
  #if NV_TENSORRT_MAJOR >= 3
  Dims shape = tensor->getDimensions();
  long sz = 1;
  for (int i=0; i<shape.nbDims; ++i) {
    sz *= shape.d[i];
  }
  return sz;
#else
  // Legacy TensorRT returns Dims3 object
  Dims3 shape = tensor->getDimensions();
  return long(shape.c) * shape.w * shape.h
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
int get_binding_size(ICudaEngine* engine, const int idx) {
#if NV_TENSORRT_MAJOR >= 3
  const Dims shape = engine->getBindingDimensions(idx);
  long sz = 1;
  for (int i=0; i<shape.nbDims; ++i) {
    sz *= shape.d[i];
  }
  return sz;
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
  
  int num_batches_;    //!< In case we need to reset the state,
public:
    /**
     * @brief Initializes time tracker.
     * @param num_batches Number of input data instances associated with
     * each element in time_tracker#batch_times_ and time_tracker#infer_times_.
     */
    time_tracker(const int num_batches) : num_batches_(num_batches) {
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
 * @brief The profiler, if enabled by a user, profiles execution times of 
 * individual layers.
 */
struct tensorrt_profiler : public IProfiler {
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

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