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

#include "logger.hpp"
#include "calibrator.hpp"

/**
 *  An inference benchmark based on NVIDIA's TensorRT library. This benchmark also measures time
 *  required to copy data to/from GPU. So, this must be pretty realistic measurements.
 */

using namespace nvinfer1;
using namespace nvcaffeparser1;


/**
 * exec <config> <model> <batch-size> <num-iters> [input_name] [output_name] [data_type]
 * https://devblogs.nvidia.com/parallelforall/deploying-deep-learning-nvidia-tensorrt/
 * https://github.com/dusty-nv/jetson-inference/blob/master/tensorNet.cpp
 */
int main(int argc, char **argv) {
  tensorrt_logger logger;
  tensorrt_calibrator calibrator;
  tensorrt_profiler profiler;
  // Define and parse commadn lien arguments
  std::string model, model_file, dtype("float32"), input_name("data"), output_name("prob"), cache_path("");
  int batch_size(1), num_warmup_batches(0), num_batches(1), report_frequency(-1);
  bool do_not_report_batch_times(false);
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
    ("cache", po::value<std::string>(&cache_path), "Path to folder that will be used to store models calibration data.")
    ("report_frequency", po::value<int>(&report_frequency), "Report performance every 'report_frequency' processed batches. "\
                                                            "Default (-1) means report in the end. For benchmarks that last not very long time "\
                                                            "this may be a good option. For very long lasting benchmarks, set this to some positive "\
                                                            "value.")
    ("no_batch_times", po::bool_switch(&do_not_report_batch_times)->default_value(false), "Do not collect and report individual batch times. You may want not "\
                                                                                          "to report individual batch times when running very long lasting benchmarks. "\
                                                                                          "Usually, it's used in combination with --report_frequency=N. If you do "\
                                                                                          "not set the report_frequency and use no_batch_times, the app will still be "\
                                                                                          "collecting batch times but will not log them.");
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
    logger.log(ILogger::Severity::kERROR, e.what());
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

  logger.log_info("[main] Creating inference builder");
  IBuilder* builder = createInferBuilder(logger);

  // Parse the caffe model to populate the network, then set the outputs.
  // For INT8 inference, the input model must be specified with 32-bit weights.
  logger.log_info("[main] Creating network and Caffe parser (model: " + model_file + ")");
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
    logger.log_info("Enabling FP16 mode");
    builder->setHalf2Mode(true);
  } else if (data_type == DataType::kINT8) {
    logger.log_info("Enabling INT8 mode");
    calibrator.setBatchSize(batch_size);

    // Allocate memory but before figure out size of input tensor.
    const nvinfer1::ITensor* input_tensor = blob_name_to_tensor->find(input_name.c_str());
    calibrator.initialize(get_tensor_size(input_tensor), 10, model, cache_path);

    builder->setInt8Mode(true);
    builder->setInt8Calibrator(&calibrator);
  } else {
    logger.log_info("Enabling FP32 mode");
  }
   
  logger.log_info("[main] Building CUDA engine");
  // This is where we need to use calibrator
  ICudaEngine* engine = builder->buildCudaEngine(*network);
    
  logger.log_info("[main] Getting network bindings");
  logger.log_bindings(engine); 
  // We need to figure out number of elements in input/output tensors.
  // Also, we need to figure out their indices.
  const auto num_bindings = engine->getNbBindings();
  const int input_index = engine->getBindingIndex(input_name.c_str()), 
            output_index = engine->getBindingIndex(output_name.c_str());
  if (input_index < 0) { logger.log_error("Input blob not found."); }
  if (output_index < 0) { logger.log_error("Output blob not found."); }
  const int input_size = get_binding_size(engine, input_index),    // Number of elements in 'data' tensor.
            output_size = get_binding_size(engine, output_index);  // Number of elements in 'prob' tensor.
    
  // Input/output data in host memory:
  std::vector<float> input(batch_size * input_size);
  std::vector<float> output(batch_size * output_size);
  fill_random(input);
    
  // Input/output data in GPU memory
  logger.log_info("[main] Filling input tensors with random data");
  std::vector<void*> buffers(num_bindings, 0);
  cudaCheck(cudaMalloc(&(buffers[input_index]), sizeof(float) * batch_size * input_size));
  cudaCheck(cudaMalloc(&(buffers[output_index]), sizeof(float) * batch_size * output_size));
  
  logger.log_info("[main] Creating execution context");
  IExecutionContext* exec_ctx = engine->createExecutionContext();
  if (vm.count("profile")) { exec_ctx->setProfiler(&profiler); }
  
  const auto num_input_bytes = sizeof(float) * input.size();
  const auto num_output_bytes = sizeof(float) * output.size();
  
  logger.log_info("[main] Running warmup iterations");
  for (int i=0; i<num_warmup_batches; ++i) {
    cudaCheck(cudaMemcpy(buffers[input_index], input.data(), num_input_bytes, cudaMemcpyHostToDevice));
    if(!exec_ctx->execute(batch_size, buffers.data())) {logger.log_error("Kernel was not run");}
    cudaCheck(cudaMemcpy(output.data(), buffers[output_index], num_output_bytes, cudaMemcpyDeviceToHost));
  }
  //
  logger.log_info("[main] Running benchmarks");
  time_tracker tm_tracker(num_batches);
  if (vm.count("profile")) { 
    profiler.reset(); 
  }
  for (int i=0; i<num_batches; ++i) {
    tm_tracker.batch_started();
    // Copy Input Data to the GPU.
    cudaCheck(cudaMemcpy(buffers[input_index], input.data(), num_input_bytes, cudaMemcpyHostToDevice));
    // Launch an instance of the GIE compute kernel.
    tm_tracker.infer_started();
    if(!exec_ctx->execute(batch_size, buffers.data())) {logger.log_error("Kernel was not run");}
    tm_tracker.infer_done();
    // Copy Output Data to the Host.
    cudaCheck(cudaMemcpy(output.data(), buffers[output_index], num_output_bytes, cudaMemcpyDeviceToHost));
    tm_tracker.batch_done();
    //
    if (report_frequency > 0 && i%report_frequency == 0) {
      logger.log_progress(tm_tracker.get_batch_times(), tm_tracker.get_iter_idx(), batch_size, "batch_");
      tm_tracker.new_iteration();
      logger.log_progress(tm_tracker.get_infer_times(), tm_tracker.get_iter_idx(), batch_size, "");
    }
  }
  
  if (vm.count("profile")) { 
    profiler.printLayerTimes(num_batches);
  }
  logger.log_info("[main] Reporting results");
  logger.log_final_results(tm_tracker.get_batch_times(), batch_size, "batch_", !do_not_report_batch_times);
  logger.log_final_results(tm_tracker.get_infer_times(), batch_size, "", !do_not_report_batch_times);
  
  logger.log_info("[main] Cleaning buffers");
  if (data_type == DataType::kINT8) {
    calibrator.freeCalibrationMemory();
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