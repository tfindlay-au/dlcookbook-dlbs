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


/**
    AlexNet:      512x3x227x227 =  301 Mb,  8 GPUs = 2.35 Gb.
    Inception3:    64x3x299x299 =   65 Mb
    ResNet50:      64x3x224x224 =   37 Mb
 
 */
using namespace nvinfer1;
using namespace nvcaffeparser1;
namespace po = boost::program_options;

void parse_command_line(int argc, char **argv,
                        po::options_description opt_desc, po::variables_map& var_map,
                        inference_engine_opts& engine_opts, image_provider_opts& data_opts,
                        logger_impl& logger);

int main(int argc, char **argv) {
    //cv::setNumThreads(1);
    //tests::image_provider_tests::test_read_c_array();
    //fast_data_provider::create_dataset("/lvol/serebrya/datasets/train/", "/lvol/serebrya/datasets/tensorrt/");
    //fast_data_provider::benchmark();
    //return 0;
    // Create one global logger.
    logger_impl logger(std::cout);
    // Parse command line arguments
    inference_engine_opts engine_opts;
    image_provider_opts data_opts;
    try {
        po::options_description opt_desc("Options");
        po::variables_map var_map;
        parse_command_line(argc, argv, opt_desc, var_map, engine_opts, data_opts, logger);
        if (var_map.count("version")) {
            std::cout << NV_GIE_MAJOR << "." << NV_GIE_MINOR << "." << NV_GIE_PATCH << std::endl;
            return 0;
        }
        if (var_map.count("help")) { 
            std::cout << "HPE Deep Learning Benchmarking Suite - TensorRT backend" << std::endl
                      << opt_desc << std::endl
                      << "TensorRT version " << NV_GIE_MAJOR << "." << NV_GIE_MINOR << "." << NV_GIE_PATCH << std::endl;
        return 0;
        }
    } catch(po::error& e) {
        logger.log_error(e.what());
        return 1;
    }
    logger.log_info(engine_opts);
    logger.log_info(data_opts);
    // Create pool of inference engines. All inference engines will be listening to data queue
    // for new inference requests. All engines will be exactly the same - model, batch size etc.
    // There will be a 1:1 mapping between GPU and inference engines.
    logger.log_info("[main                  ]: Creating mGPU inference engine");
    mgpu_inference_engine engine(engine_opts, logger);
    const size_t num_engines = engine.num_engines();
    // Create pool of available task request objects. These objects (infer_task) will be initialized
    // to store input/output tensors so there will be no need to do memory allocations during benchmark.
    const float est_mp_mem = static_cast<float>(engine_opts.inference_queue_size_*(8+4+4+4+engine.input_size()*4+engine.output_size()*4)) / (1024*1024);
    logger.log_info("[main                  ]: Creating inference message pool with " + S(engine_opts.inference_queue_size_) + " messages, estimated memory is " + std::to_string(est_mp_mem) + " mb.");
    inference_msg_pool infer_msg_pool(engine_opts.inference_queue_size_, engine.batch_size(), engine.input_size(), engine.output_size(), true);
    // Create data provider. The data provider will spawn at least one thread. It will fetch free task objects
    // from pool of task objects, will populate them with data and will submit tasks to data queue. All
    // preprocessing logic needs to be implemented in data provider.
    data_provider* dataset(nullptr);
    if (data_opts.data_dir_ == "") {
        dataset = new synthetic_data_provider(&infer_msg_pool, engine.request_queue());
        logger.log_info("[main                  ]: Will use synthetic data set");
    } else {
        logger.log_info("[main                  ]: Will use real data set (" + data_opts.data_dir_ + ")");
        logger.log_warning("[main                  ]: Computing resize dimensions assuming input data has shape [BatchSize, 3, H, W] where H == W.");
        data_opts.height_ = data_opts.width_ = std::sqrt(engine.input_size() / (engine_opts.batch_size_ * 3));
        if (data_opts.data_name_ == "images") {
            dataset = new image_provider(data_opts, &infer_msg_pool, engine.request_queue(), logger);
        } else {
            dataset = new fast_data_provider(data_opts, &infer_msg_pool, engine.request_queue(), logger);
        }
    }
    logger.log_info("[main                  ]: Starting dataset threads");
    dataset->start();
    // Start pool of inference engines. This will start one thread per engine. Individual inference engines
    // will be fetching data from data queue, will be doing inference and will be submitting same task request
    // objects with inference results and statistics to decision queue.
    logger.log_info("[main                  ]: Starting engine threads");
    engine.start();
  
    logger.log_info("[main                  ]: Running warmup iterations");
    for (size_t i=0; i<engine_opts.num_warmup_batches_; ++i) {
        // Do warmup iterations. Just fetch inference results from decision queue
        // and put them back to pool of free task objects. All data preprocessing,
        // submission and classification are done in backgroud threads.
        for (size_t j=0; j<num_engines; ++j) {
            inference_msg *msg = engine.response_queue()->pop();
            infer_msg_pool.release(msg);
        }
    }
    // This reset will not happen immidiately, but next time an engine processes a batch.
    // So, they may reset their states at slightly different moments.
    engine.reset();

    logger.log_info("[main                  ]: Running benchmarks");
    time_tracker tm_tracker(engine_opts.num_batches_);
    for (size_t i=0; i<engine_opts.num_batches_; ++i) {
        tm_tracker.batch_started();
        for (size_t j=0; j<num_engines; ++j) {
            inference_msg *msg = engine.response_queue()->pop();
            infer_msg_pool.release(msg);
        }
        tm_tracker.batch_done();
        if (engine_opts.report_frequency_ > 0 && i>0 && i%engine_opts.report_frequency_ == 0) {
            logger.log_progress(tm_tracker.get_batch_times(), tm_tracker.get_iter_idx(), engine_opts.batch_size_, "total_");
            tm_tracker.new_iteration();
        }
    }
    // Shutdown everything and wait for all threads to exit.
    logger.log_info("[main                  ]: Stopping and joining threads");
    dataset->stop();  engine.stop();  infer_msg_pool.close();
    logger.log_info("[main                  ]: Waiting for data provider ...");
    dataset->join();
    logger.log_info("[main                  ]: Waiting for inference engine ...");
    engine.join();
    delete  dataset;  dataset = nullptr;
    // Log final results.
    logger.log_info("[main                  ]: Reporting results");
    for (size_t i=0; i<num_engines; ++i) {
        time_tracker *tracker = engine.engine(i)->get_time_tracker();
        const std::string gpu_id = std::to_string(engine.engine(i)->engine_id());
        logger.log_final_results(tracker->get_batch_times(), engine_opts.batch_size_, "gpu_" + gpu_id + "_infer_", !engine_opts.do_not_report_batch_times_);
        logger.log_final_results(tracker->get_batch_times(), engine_opts.batch_size_, "gpu_" + gpu_id + "_batch_", !engine_opts.do_not_report_batch_times_);
    }
    logger.log_final_results(tm_tracker.get_batch_times(), engine_opts.batch_size_*num_engines, "total_", false);
    logger.log_final_results(tm_tracker.get_batch_times(), engine_opts.batch_size_*num_engines, "", !engine_opts.do_not_report_batch_times_);
    return 0;
}

void parse_command_line(int argc, char **argv,
                        boost::program_options::options_description opt_desc, po::variables_map& var_map,
                        inference_engine_opts& engine_opts, image_provider_opts& data_opts,
                        logger_impl& logger) {
    namespace po = boost::program_options;
    std::string gpus;
    int batch_size(static_cast<int>(engine_opts.batch_size_)),
        num_warmup_batches(static_cast<int>(engine_opts.num_warmup_batches_)),
        num_batches(static_cast<int>(engine_opts.num_batches_)),
        report_frequency(static_cast<int>(engine_opts.report_frequency_)),
        inference_queue_size(static_cast<int>(engine_opts.inference_queue_size_));
    int num_prefetchers(static_cast<int>(data_opts.num_prefetchers_)),
        num_decoders(static_cast<int>(data_opts.num_decoders_)),
        prefetch_queue_size(static_cast<int>(data_opts.prefetch_queue_size_)),
        prefetch_batch_size(static_cast<int>(data_opts.prefetch_batch_size_));

    opt_desc.add_options()
        ("help", "Print help message")
        ("version", "Print version")
        ("gpus", po::value<std::string>(&gpus), "A comma seperated list of GPU identifiers to use.")
        ("model", po::value<std::string>(&engine_opts.model_id_), "Model identifier like alexnet, resent18 etc. Used to store calibration caches.")
        ("model_file", po::value<std::string>(&engine_opts.model_file_), "Caffe's prototxt deploy (inference) model.")
        ("batch_size", po::value<int>(&batch_size), "Per device batch size.")
        ("dtype", po::value<std::string>(&engine_opts.dtype_), "Type of data variables: float(same as float32), float32, float16 or int8.")
        ("num_warmup_batches", po::value<int>(&num_warmup_batches), "Number of warmup iterations.")
        ("num_batches", po::value<int>(&num_batches), "Number of benchmark iterations.")
        ("profile",  po::bool_switch(&engine_opts.use_profiler_)->default_value(false), "Profile model and report results.")
        ("input", po::value<std::string>(&engine_opts.input_name_), "Name of an input data tensor (data).")
        ("output", po::value<std::string>(&engine_opts.output_name_), "Name of an output data tensor (prob).")
        ("cache", po::value<std::string>(&engine_opts.calibrator_cache_path_), "Path to folder that will be used to store models calibration data.")
        ("report_frequency", po::value<int>(&report_frequency),
            "Report performance every 'report_frequency' processed batches. "\
            "Default (-1) means report in the end. For benchmarks that last not very long time "\
            "this may be a good option. For very long lasting benchmarks, set this to some positive "\
            "value.")
        ("no_batch_times", po::bool_switch(&engine_opts.do_not_report_batch_times_)->default_value(false),
            "Do not collect and report individual batch times. You may want not "\
            "to report individual batch times when running very long lasting benchmarks. "\
            "Usually, it's used in combination with --report_frequency=N. If you do "\
            "not set the report_frequency and use no_batch_times, the app will still be "\
            "collecting batch times but will not log them.")
        ("data_dir", po::value<std::string>(&data_opts.data_dir_), "Path to a dataset.")
        ("data_name", po::value<std::string>(&data_opts.data_name_), "Name of a dataset - 'images' or 'tensors'.")
        ("resize_method", po::value<std::string>(&data_opts.resize_method_), "How to resize images: 'crop' or 'resize'.")
        ("num_prefetchers", po::value<int>(&num_prefetchers), "Number of prefetch threads (data readers).")
        ("prefetch_queue_size", po::value<int>(&prefetch_queue_size), "Number of batches to prefetch.")
        ("prefetch_batch_size", po::value<int>(&prefetch_batch_size), "Size of a prefetch batch.")
        ("num_decoders", po::value<int>(&num_decoders), "Number of decoder threads (that convert JPEG to input blobs).")
        ("fake_decoder",  po::bool_switch(&data_opts.fake_decoder_)->default_value(false),
            "If set, fake decoder will be used. Fake decoder is a decoder that does not decode JPEG images into "\
            "different representation, but just passes through itself inference requests. This option is useful "\
            "to benchmark prefetchers and/or storage.")
        ("inference_queue_size", po::value<int>(&inference_queue_size), "Number of pre-allocated inference requests.")
        ("fake_inference",  po::bool_switch(&engine_opts.fake_inference_)->default_value(false));
   
    po::store(po::parse_command_line(argc, argv, opt_desc), var_map);
    if (var_map.count("version") > 0 || var_map.count("help") > 0)
        return;
    po::notify(var_map);
    
    if (batch_size <= 0)
        throw po::error("Batch size must be strictly positive (size=" + std::to_string(batch_size) + ")");
    if (inference_queue_size <= 0)
        throw po::error("Inference queue size must be strictly positive (size=" + std::to_string(inference_queue_size) + ")");
    engine_opts.batch_size_ = static_cast<size_t>(batch_size);
    engine_opts.inference_queue_size_ = static_cast<size_t>(inference_queue_size);
    engine_opts.num_warmup_batches_ = static_cast<size_t>(std::max(num_warmup_batches, 0));
    engine_opts.num_batches_ = static_cast<size_t>(std::max(num_batches, 0));
    engine_opts.report_frequency_ = static_cast<size_t>(std::max(report_frequency, 0));

    engine_opts.gpus_.clear();
    std::replace(gpus.begin(), gpus.end(), ',', ' ');
    std::istringstream stream(gpus);
    int gpu_id = 0;
    while (stream>>gpu_id) { 
        engine_opts.gpus_.push_back(gpu_id);
        logger.log_info("Will use GPU: " + std::to_string(gpu_id));
    }
    
    if (engine_opts.fake_inference_ && engine_opts.gpus_.size() > 1)
        logger.log_warning("Fake inference will be used but number of engines is > 1. You may want to set it to 1.");
 
    if (data_opts.data_dir_ != "") {
        if (num_prefetchers <= 0)      num_prefetchers = 3 * engine_opts.gpus_.size();
        if (num_decoders <= 0)         num_decoders = 3 * engine_opts.gpus_.size();
        if (prefetch_queue_size <= 0)  prefetch_queue_size = 3 * engine_opts.gpus_.size();
        if (prefetch_batch_size <= 0)  prefetch_batch_size = engine_opts.batch_size_;
    }
    data_opts.num_prefetchers_ = static_cast<size_t>(std::max(num_prefetchers, 0));
    data_opts.num_decoders_ = static_cast<size_t>(std::max(num_decoders, 0));
    data_opts.prefetch_queue_size_ = static_cast<size_t>(std::max(prefetch_queue_size, 0));
    data_opts.prefetch_batch_size_ = static_cast<size_t>(std::max(prefetch_batch_size, 0));;
    if (data_opts.fake_decoder_ && data_opts.num_decoders_ > 1)
        logger.log_warning("Fake decoder will be used but number of decoders > 1. You may want to set it to 1.");
}
