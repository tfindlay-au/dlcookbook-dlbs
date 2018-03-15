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
/**
   Completely based on giexec.cpp example in /usr/src/gie_samples/samples/giexec/
 */
#include <iostream>
#include <string>
#include <random>
#include <map>
#include <ios>
#include <functional>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <fstream>
#include <ctime>

#include <string.h>

#include <cuda_runtime_api.h>
//#include <cuda_fp16.h>

#include <NvCaffeParser.h>
#include <NvInfer.h>

#include <boost/program_options.hpp>

/**
 *  An inference benchmark based on NVIDIA's TensorRT library. This benchmark also measures time
 *  required to copy data to/from GPU. So, this must be pretty realistic measurements.
 */

using namespace nvinfer1;
using namespace nvcaffeparser1;

// Check CUDA result.
#define cudaCheck(ans) { cudaCheckf((ans), __FILE__, __LINE__); }
inline void cudaCheckf(const cudaError_t code, const char *file, const int line, const bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Get environmental variables.
bool get_boolean_env_var(const std::string& name, bool default_value);
std::string get_string_env_var(const std::string& name, const std::string& default_value);

// Fill vector with random numebrs uniformly dsitributed in [0, 1).
void fill_random(std::vector<float>& vec);

// A simple timer implementation.
class timer {
public:
  timer() {
    restart();
  }
  void restart() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  float ms_elapsed() const {
    const auto now = std::chrono::high_resolution_clock::now();
    return float(std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count()) / 1000.0;
  }
private:
  std::chrono::high_resolution_clock::time_point start_;
};

//A simple logger for TensorRT library.
class my_logger : public ILogger {
public:
  /**
   * severity: [kINTERNAL_ERROR, kERROR, kWARNING, kINFO]
   */
  virtual void log(Severity severity, const char* msg) override {
    std::cerr << time_stamp() << " " 
              << log_levels_[severity] << " " 
              << msg << std::endl;
    if (severity == ILogger::Severity::kINTERNAL_ERROR || severity == ILogger::Severity::kERROR) {
      exit(1);
    }
  }
  void log_internal_error(const char* msg) { log(ILogger::Severity::kINTERNAL_ERROR, msg); }
  void log_error(const char* msg) { log(ILogger::Severity::kERROR, msg); }
  void log_warning(const char* msg) { log(ILogger::Severity::kWARNING, msg); }
  void log_info(const char* msg) { log(ILogger::Severity::kINFO, msg); }
  
  void log_internal_error(const std::string& msg) { log_internal_error(msg.c_str()); }
  void log_error(const std::string& msg) { log_error(msg.c_str()); }
  void log_warning(const std::string& msg) { log_warning(msg.c_str()); }
  void log_info(const std::string& msg) { log_info(msg.c_str()); }
private:
  std::string time_stamp() {
    time_t rawtime;
    time (&rawtime);
    struct tm * timeinfo = localtime(&rawtime);
    // YYYY-mm-dd HH:MM:SS    19 characters
    char buffer[20];
    const auto len = strftime(buffer,sizeof(buffer),"%F %T",timeinfo);
    return (len > 0 ? std::string(buffer) : std::string(19, ' '));
  }
private:
  std::map<Severity, std::string> log_levels_ = {
    {ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR"},
    {ILogger::Severity::kERROR,          "         ERROR"},
    {ILogger::Severity::kWARNING,        "       WARNING"},
    {ILogger::Severity::kINFO,           "          INFO"}
  };
  
};
my_logger g_logger;
//
struct profiler : public IProfiler {
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

} g_profiler;

/**
 * TODO: Is it true that we can run first all networks with small batch sizes to create fake calibration 
 *       caches and then just load that. The problem is that with large batch size calibration memory takes
 *       too much memory, for instance, an input image of 3x255x255x512 = 381 MB where 512 is a batch size.
 */
class calibrator : public IInt8Calibrator {
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
  void initialize(const Dims3 input_shape, const int num_batches, const std::string& model, const std::string& calibration_cache_path) {
    std::ostringstream stream;
    stream << "[calibrator] Calibrator::initialize(input_shape=[" << input_shape.c << ", input_shape.h" << ", input_shape.w]"
           << ", num_batches=" << num_batches << ", model = " << model << ")";
    log(stream.str());

    input_size_ = input_shape.c * input_shape.w * input_shape.h;
    num_batches_ = num_batches;
    next_batch_ = 0;

    calibration_cache_path_ = calibration_cache_path;
    model_ = model;

    if (calibration_cache_path_ == "") {
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
    if (calibration_cache_path_ != "" && model_ != "") {
      return calibration_cache_path_ + "/" + model_ + "_" + suffix + ".bin";
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
  int input_size_ = 0;                  // Size of one instance (multiplication of all dimensions)
  int num_batches_ = 0;                 // Number of batches to use for calibration
  int next_batch_ = 0;                  // During calibration, index of the next batch
  std::vector<float> host_batch_;       // Batch data in host memory
  void *gpu_batch_ = nullptr;           // Batch data in GPU memory
  
  double quantile_ = 0.5;
  double cutoff_ = 0.5;
  
  bool do_log_ = true;
  
  std::string calibration_cache_path_;  // Path to calibration cache.
  std::string model_;                   // Neural network model (to save/load calibration caches).
  char* calibration_cache_ = nullptr;   // Calibration cache loaded from file.
  size_t calibration_cache_length_ = 0;
  char* histogram_cache_ = nullptr;     // Histogram cache loaded from file.
  size_t histogram_cache_length_ = 0;
} g_calibrator;

// Get number of elements in this tensor (blob).
int get_binding_size(ICudaEngine* engine, const int idx);

// Print engine bindings (input/output blobs)
void report_bindings(ICudaEngine* engine);

// Compute and print results. At the exit, vec will be sorted.
void report_results(std::vector<float>& vec, const std::string& name_prefix, const int batch_size);

/**
 * exec <config> <model> <batch-size> <num-iters> [input_name] [output_name] [data_type]
 * https://devblogs.nvidia.com/parallelforall/deploying-deep-learning-nvidia-tensorrt/
 * https://github.com/dusty-nv/jetson-inference/blob/master/tensorNet.cpp
 */
int main(int argc, char **argv) {
  // Define and parse commadn lien arguments
  std::string model, model_file, dtype("float32"), input_name("data"), output_name("prob"), calibration_cache_path("");
  int batch_size(1), num_warmup_batches(0), num_batches(1);
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
    ("help",  "Print help message")
    ("version",  "Print version")
    ("model", po::value<std::string>(&model), "Model identifier like alexnet, resent18 etc. Used to store calibration caches.")
    ("model_file", po::value<std::string>(&model_file)->required(), "Caffe's prototxt deploy (inference) model.")
    ("batch_size", po::value<int>(&batch_size), "Per device batch size.")
    ("dtype", po::value<std::string>(&dtype), "Type of data variables: float(same as float32), float32, float16 or int8.")
    ("num_warmup_batches", po::value<int>(&num_warmup_batches), "Number of warmup iterations.")
    ("num_batches", po::value<int>(&num_batches), "Number of benchmark iterations.")
    ("profile",  "Profile model and report results.")
    ("input", po::value<std::string>(&input_name), "Name of an input data tensor (data).")
    ("output", po::value<std::string>(&output_name), "Name of an output data tensor (prob).")
    ("calibration_cache_path", po::value<std::string>(&calibration_cache_path), "Path to folder that will be used to store models calibration data.");
  po::variables_map vm;
  
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("version")) {
      std::cout << NV_GIE_MAJOR << "." << NV_GIE_MINOR << "." << NV_GIE_PATCH << std::endl;
      return 0;
    }
    if (vm.count("help")) { 
      std::cout << "TensorRT Benchmarks" << std::endl
                << desc << std::endl
                << "version 2.0.2 (nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64)" << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch(po::error& e) {
    g_logger.log(ILogger::Severity::kERROR, e.what());
    std::cerr << desc << std::endl;
    return 1;
  }

  // Figure out type of data to work with
  if (dtype == "float") {
    dtype = "float32";
  }
  const DataType data_type = (
    dtype == "float32" ? DataType::kFLOAT : 
                         (dtype == "float16" ? DataType::kHALF : 
                                               DataType::kINT8)
  );

  g_logger.log_info("[main] Creating inference builder");
  IBuilder* builder = createInferBuilder(g_logger);

  // Parse the caffe model to populate the network, then set the outputs.
  // For INT8 inference, the input model must be specified with 32-bit weights.
  g_logger.log_info("[main] Creating network and Caffe parser (model: " + model_file + ")");
  INetworkDefinition* network = builder->createNetwork();
  ICaffeParser* caffe_parser = createCaffeParser();
  const IBlobNameToTensor* blob_name_to_tensor = caffe_parser->parse(
    model_file.c_str(), // *.prototxt caffe model definition
    nullptr,       // if null, random weights?
    *network, 
    (data_type == DataType::kINT8 ? DataType::kFLOAT : data_type)
  );
    
  // Specify what tensors are output tensors.
  network->markOutput(*blob_name_to_tensor->find(output_name.c_str()));

  // Build the engine.
  builder->setMaxBatchSize(batch_size);
  builder->setMaxWorkspaceSize(1 << 30); 
  // Half and INT8 precision specific options
  if (data_type == DataType::kHALF) {
    g_logger.log_info("Enabling FP16 mode");
    builder->setHalf2Mode(true);
  } else if (data_type == DataType::kINT8) {
    g_logger.log_info("Enabling INT8 mode");
    g_calibrator.setBatchSize(batch_size);

    // Allocate memory but before figure out size of input tensor.
    const nvinfer1::ITensor* input_tensor = blob_name_to_tensor->find(input_name.c_str());
    g_calibrator.initialize(
      input_tensor->getDimensions(),
      10,
      model,
      calibration_cache_path
    );

    builder->setInt8Mode(true);
    builder->setInt8Calibrator(&g_calibrator);
  } else {
    g_logger.log_info("Enabling FP32 mode");
  }
   
  g_logger.log_info("[main] Building CUDA engine");
  // This is where we need to use calibrator
  ICudaEngine* engine = builder->buildCudaEngine(*network);
    
  g_logger.log_info("[main] Getting network bindings");
  report_bindings(engine); 
  // We need to figure out number of elements in input/output tensors.
  // Also, we need to figure out their indices.
  const auto num_bindings = engine->getNbBindings();
  const int input_index = engine->getBindingIndex(input_name.c_str()), 
            output_index = engine->getBindingIndex(output_name.c_str());
  if (input_index < 0) { g_logger.log_error("Input blob not found."); }
  if (output_index < 0) { g_logger.log_error("Output blob not found."); }
  const int input_size = get_binding_size(engine, input_index),    // Number of elements in 'data' tensor.
            output_size = get_binding_size(engine, output_index);  // Number of elements in 'prob' tensor.
    
  // Input/output data in host memory:
  std::vector<float> input(batch_size * input_size);
  std::vector<float> output(batch_size * output_size);
  fill_random(input);
    
  // Input/output data in GPU memory
  g_logger.log_info("[main] Filling input tensors with random data");
  std::vector<void*> buffers(num_bindings, 0);
  cudaCheck(cudaMalloc(&(buffers[input_index]), sizeof(float) * batch_size * input_size));
  cudaCheck(cudaMalloc(&(buffers[output_index]), sizeof(float) * batch_size * output_size));
  
  g_logger.log_info("[main] Creating execution context");
  IExecutionContext* exec_ctx = engine->createExecutionContext();
  if (vm.count("profile")) { exec_ctx->setProfiler(&g_profiler); }
  
  const auto num_input_bytes = sizeof(float) * input.size();
  const auto num_output_bytes = sizeof(float) * output.size();
  
  g_logger.log_info("[main] Running warmup iterations");
  for (int i=0; i<num_warmup_batches; ++i) {
    cudaCheck(cudaMemcpy(buffers[input_index], input.data(), num_input_bytes, cudaMemcpyHostToDevice));
    if(!exec_ctx->execute(batch_size, buffers.data())) {g_logger.log_error("Kernel was not run");}
    cudaCheck(cudaMemcpy(output.data(), buffers[output_index], num_output_bytes, cudaMemcpyDeviceToHost));
  }
  
  std::vector<float> total(num_batches, 0);
  std::vector<float> inference(num_batches, 0);
  g_logger.log_info("[main] Running benchmarks");
  timer total_timer, inference_timer;
  if (vm.count("profile")) { 
    g_profiler.reset(); 
  }
  for (int i=0; i<num_batches; ++i) {
    total_timer.restart();
    // Copy Input Data to the GPU.
    cudaCheck(cudaMemcpy(buffers[input_index], input.data(), num_input_bytes, cudaMemcpyHostToDevice));
    // Launch an instance of the GIE compute kernel.
    inference_timer.restart();
    if(!exec_ctx->execute(batch_size, buffers.data())) {g_logger.log_error("Kernel was not run");}
    inference[i] = inference_timer.ms_elapsed();
    // Copy Output Data to the Host.
    cudaCheck(cudaMemcpy(output.data(), buffers[output_index], num_output_bytes, cudaMemcpyDeviceToHost));
    total[i] += total_timer.ms_elapsed();
  }
  
  if (vm.count("profile")) { 
    g_profiler.printLayerTimes(num_batches);
  }
  g_logger.log_info("[main] Reporting results");
  report_results(total, "total_", batch_size);           // Total, true, time including data transfers.
  report_results(inference, "", batch_size);             // Pure inference time.
  
  g_logger.log_info("[main] Cleaning buffers");
  if (data_type == DataType::kINT8) {
    g_calibrator.freeCalibrationMemory();
  }
  cudaFree(buffers[output_index]);
  cudaFree(buffers[input_index]);
  exec_ctx->destroy();
  network->destroy();
  caffe_parser->destroy();
  engine->destroy();
  builder->destroy();

  return 0;
}

bool get_boolean_env_var(const std::string& name, bool default_value) {
  const char* buffer = getenv(name.c_str());
  if (buffer == nullptr) {
    return default_value;
  }
  const std::string value(buffer);
  if (value == "true" || value == "1") {
    return true;
  }
  return false;
}

std::string get_string_env_var(const std::string& name, const std::string& default_value) {
  const char* buffer = getenv(name.c_str());
  if (buffer == nullptr) {
    return default_value;
  }
  return std::string(buffer);
}

int get_binding_size(ICudaEngine* engine, const int idx) {
  const Dims3 dims = engine->getBindingDimensions(idx);
  return dims.c * dims.h * dims.w;
}

void fill_random(std::vector<float>& vec) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(std::begin(vec), std::end(vec), gen);
}

void report_bindings(ICudaEngine* engine) {
  const auto num_bindings = engine->getNbBindings();
  std::cout << "engine::number of bindings = " << num_bindings << std::endl;
  for (auto i=0; i<num_bindings; ++i) {
    std::cout << "engine::binding index = " << i << ", name = " << engine->getBindingName(i) << ", is input = " << engine->bindingIsInput(i);
    const Dims3 dims = engine->getBindingDimensions(i);
    std::cout << ", shape = " << dims.c << "," << dims.h << "," << dims.w << std::endl;
  }
}

void report_results(std::vector<float>& v, const std::string& name_prefix, const int batch_size) {
  std::sort(v.begin(), v.end());
  const float sum = std::accumulate(v.begin(), v.end(), 0.0f);
  const float mean = sum / v.size();
  const float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0f);
  const float stdev = std::sqrt(sq_sum / v.size() - mean * mean);
  const float throughput = 1000.0 * batch_size / mean;
  
  std::cout << "__results." << name_prefix << "time__= " << mean  << std::endl;
  std::cout << "__results." << name_prefix << "throughput__= " << throughput  << std::endl;
  std::cout << "__results." << name_prefix << "time_data__=[";
  for (int i=0; i<v.size(); ++i) {
    if (i != 0) { std::cout << ","; }
    std::cout << v[i];
  }
  std::cout << "]" << std::endl;
  
  std::cout << "__results." << name_prefix << "time_stdev__= " << stdev  << std::endl;
  std::cout << "__results." << name_prefix << "time_min__= " << v.front()  << std::endl;
  std::cout << "__results." << name_prefix << "time_max__= " << v.back()  << std::endl;
}

// Asynch version:
/*
 *     cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int i=0; i<num_iters; ++i) {
      // Copy Input Data to the GPU
      cudaCheck(cudaMemcpyAsync(buffers[input_index], input.data(), sizeof(float) * batch_size * input_size, cudaMemcpyHostToDevice, stream));
      // Launch an instance of the GIE compute kernel
      if(!exec_ctx->enqueue(batch_size, buffers.data(), stream, nullptr)) {
        g_logger.log(ILogger::Severity::kERROR, "Kernel was not enqueued");
      }
      // Copy Output Data to the Host
      cudaCheck(cudaMemcpyAsync(output.data(), buffers[output_index], sizeof(float) * batch_size * output_size, cudaMemcpyDeviceToHost, stream));
      //
      cudaCheck(cudaStreamSynchronize(stream));
    }
*/