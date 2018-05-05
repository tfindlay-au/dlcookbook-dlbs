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
#include <thread>
#include <ctime>

#include <string.h>

#include <cuda_runtime_api.h>

#include <boost/program_options.hpp>

#include "logger.hpp"
#include "infer_engine.hpp"
#include "data_providers.hpp"

/**
 *  @brief An inference benchmark based on NVIDIA's TensorRT library.
 * 
 *  The TensorRT backend benchmarks inference with one or several GPUs. If several
 *  GPUs are provided, each GPU independently runs its own instance of inference engine.
 *  Each GPU runs exactly the same configuration - model, batch size etc. As opposed to
 *  training benchmarks, batch size does not change as number of GPUs changes. It's always
 *  per GPU batch size. 
 * 
 *  This backend provides performance results at three different levels:
 *    - Inference time is the pure inference time without any overhead associated with data ingestion
 *                including CPU <--> GPU transfers. This is upper bound for the performance. We do not
 *                prestage tensors now so it all happens sequentially:
 *                     * Copy data from CPU to GPU memory
 *                     * Run inference
 *                     * Copy results from GPU to CPU memory
 *    - Batch time is the time that includes inference time and time required to copy data to/form GPU.
 *    - Actual time that includes inference time, time associated with CPU-GPU data transfers and time
 *                associated with entire data ingestion pipeline. Copying data from storage into CPU
 *                memory, preprocessing and any other overhead is taken into account. Not all is done
 *                sequentially. In particular, data ingestion into request queue is done in multiple background
 *                threads simoultaneously with GPU computations. This time is useful to benchmark preprocessing
 *                efficiency and storage.
 */

using namespace nvinfer1;
using namespace nvcaffeparser1;

int main(int argc, char **argv) {
  tests::raw_img_data_provider_tests::read_image();
  return 0;
  tensorrt_logger logger;

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
                << "version " << NV_GIE_MAJOR << "." << NV_GIE_MINOR << "." << NV_GIE_PATCH << std::endl;
      return 0;
    }
    po::notify(vm);
  } catch(po::error& e) {
    logger.log(ILogger::Severity::kERROR, e.what());
    std::cerr << desc << std::endl;
    return 1;
  }

  // Create two concurrent queues, one for inference requests (data_queue) and one
  // for results (decision_queue). These queue objects do not spawn any threads.
  thread_safe_queue<infer_task*> data_queue, decision_queue;
  // Create pool of inference engines. All inference engines will be listening to data queue
  // for new inference requests. All engines will be exactly the same - model, batch size etc.
  // There will be a 1:1 mapping between GPU and inference engines.
  infer_engine_pool engine_pool("0", logger, model, model_file, dtype, batch_size, num_batches,
                                vm.count("profile") > 0, output_name, input_name, cache_path);
  // Create pool of available task request objects. These objects (infer_task) will be initialized
  // to store input/output tensors so there will be no need to do memory allocations during benchmark.
  task_pool<infer_task> task_pool(10, engine_pool.input_size(), engine_pool.output_size(), true);
  // Create data provider. The data provider will spawn at least one thread. It will fetch free task objects
  // from pool of task objects, will populate them with data and will submit tasks to data queue. All
  // preprocessing logic needs to be implemented in data provider.
  synthetic_data_provider<infer_task> data_provider(&task_pool, &data_queue);
  data_provider.start();
  // Start pool of inference engines. This will start one thread per engine. Individual inference engines
  // will be fetching data from data queue, will be doing inference and will be submitting same task request
  // objects with inference results and statistics to decision queue.
  engine_pool.start(data_queue, decision_queue);
  
  const int num_engines = engine_pool.num_engines();
  logger.log_info("[main] Running warmup iterations");
  for (int i=0; i<num_warmup_batches; ++i) {
      // Do warmup iterations. Just fetch inference results from decision queue
      // and put them back to pool of free task objects. All data preprocessing,
      // submission and classification are done in backgroud threads.
      for (int i=0; i<num_engines; ++i)
        task_pool.release(decision_queue.pop());
  }
  if (engine_pool.layer_wise_profiling()) {
      // This reset will not happen immidiately, but next time an engine processes a batch.
      // So, they may reset their states at slightly different moments.
      engine_pool.reset();
  }
  logger.log_info("[main] Running benchmarks");
  time_tracker tm_tracker(num_batches);
  for (int i=0; i<num_batches; ++i) {
    tm_tracker.batch_started();
    for (int i=0; i<num_engines; ++i)
        task_pool.release(decision_queue.pop());
    tm_tracker.batch_done();
    if (report_frequency > 0 && i>0 && i%report_frequency == 0) {
      logger.log_progress(tm_tracker.get_batch_times(), tm_tracker.get_iter_idx(), batch_size, "total_");
      tm_tracker.new_iteration();
    }
  }
  // Shutdown everything and wait for all threads to exit.
  task_pool.stop();
  data_provider.stop();
  engine_pool.stop();
  data_provider.join();
  engine_pool.join();
  // Log final results.
  logger.log_info("[main] Reporting results");
  for (int i=0; i<num_engines; ++i) {
      time_tracker *tracker = engine_pool.engine(i)->get_time_tracker();
      const std::string gpu_id = std::to_string(engine_pool.engine(i)->gpu_id());
      logger.log_final_results(tracker->get_batch_times(), batch_size, "gpu_" + gpu_id + "_infer_", !do_not_report_batch_times);
      logger.log_final_results(tracker->get_batch_times(), batch_size, "gpu_" + gpu_id + "_batch_", !do_not_report_batch_times);
  }
  logger.log_final_results(tm_tracker.get_batch_times(), batch_size*num_engines, "total_", false);
  logger.log_final_results(tm_tracker.get_batch_times(), batch_size*num_engines, "", !do_not_report_batch_times);
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