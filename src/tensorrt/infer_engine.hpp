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
 * We need the following:
 *    1. Data provider queues. Two types of queues are supported - with real
 *       data and with fake data (infinite  queue)
 *    2. A thread that will pop from the queue and will process that data on
 *       some particular GPU.
 *    3. How do we return a result?
 */
#ifndef DLBS_TENSORRT_BACKEND_INFER_ENGINE
#define DLBS_TENSORRT_BACKEND_INFER_ENGINE

#include <sstream>
#include "infer_msg.hpp"
#include "calibrator.hpp"
#include <NvCaffeParser.h>

/**
 * @brief Options to instantiate an inference engine.
 */
struct inference_engine_opts {
    std::string model_file_;               //!< Full path to a Caffe's protobuf inference descriptor.
    std::string dtype_="float32";          //!< Data type (precision): float32(float), float16, int8.
    int batch_size_;                       //!< Batch size.
    int num_warmup_batches_;               //!< Number of warmup batches.
    int num_batches_;                      //!< Number of benchmark batches.
    bool use_profiler_=false;              //!< Perform layer-wise model profiling.
    std::string input_name_="data";        //!< Name of an input blob.
    std::string output_name_="prob";       //!< Name of an output blob.
    
    std::string model_id_;                 //!< Model ID, used to store calibrator cache if data type is int8.
    std::string calibrator_cache_path_=""; //!< Path to store calibrator cache if data type is int8.
    
    DataType data_type() const {
        if (dtype_ == "float32" || dtype_ == "float")
            return DataType::kFLOAT;
        if (dtype_ == "float16")
            return DataType::kHALF;
        return DataType::kINT8;
    }
    
    void log() {
        std::cout << "[inference_engine_opts] " 
                  << "model_file=" << model_file_ << ", dtype=" << dtype_ << ", batch_size=" << batch_size_
                  << ", num_warmup_batches=" << num_warmup_batches_ << ", num_batches=" << num_batches_
                  << ", use_profiler=" << use_profiler_ << ", input_blob_name=" << input_name_
                  << ", output_blob_name=" << output_name_ << ", model_id=" << model_id_
                  << ", calibrator_cache_path=" << calibrator_cache_path_
                  << std::endl;
    }
};

/**
 * @brief TensorRT inference engine that works with one GPU.
 * 
 * An engine can operate in two regimes - synchronous and asynchronous. Synchronous
 * regime is useful when users want to submit inference requests on their own. In asynchronous
 * regime, an inference engine runs in separate thread fetching inference request from an input
 * request queue. 
 * 
 * To use multiple inference engines with multiplt GPUs, use mgpu_inference_engine instead.
 * 
 * Synchronous regime (can only be used by one caller thread).
 * @code{.cpp}
 *     inference_engine engine (...);                             // Instantiate inference engine.
 *     inference_msg* task = engine.new_inferene_message(...);    // Create inference task allocating memory for input/output host blobs.
 *     msg->input_ ...;                                           // Fill in input blob
 *     engine.infer(msg);                                         // Run inference. The 'msg' can be reused for subsequent calls.
 *     msg->output_ ...;                                          // Get inference output resutls.
 *     delete msg;                                                // Deallocate memory.
 * @endcode
 * 
 * Asynchronous regime.
 * @code
 *     inference_engine engine (...);                                         // Instantiate inference engine.
 *     abstract_queue<inference_msg*>& request_queue = get_request_queue();   // Create input request queue.
 *     abstract_queue<inference_msg*>& response_queue = get_response_queue(); // Create output response queue.
 *     engine.start(request_queue, response_queue);                           // Run engine in backgroun thread.
 *     while (true) {
 *         inference_msg* msg = response_queue.pop();                         // Fetch inference response
 *         msg->output_ ...;                                                  // Process response.
 *         delete msg;                                                        // If inference message pool is not used, deallocate memory.      
 *     }
 * @endcode
 */
class inference_engine {
private:
    int gpu_id_;
    tensorrt_logger& logger_;
    tensorrt_calibrator calibrator_;
    tensorrt_profiler *profiler_ = nullptr;
    time_tracker tm_tracker_;
    
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* exec_ctx_ = nullptr;
    
    long nbatches_ = 0;
    
    int batch_sz_;
    int input_sz_ = 0;
    int output_sz_ = 0;
    
    int input_idx_ = 0;
    int output_idx_ = 0;
    
    std::vector<void*> dev_buf_;
    std::atomic_bool stop_;
    std::atomic_bool reset_;
 
    std::thread *internal_thread_;
private:
    void init_device_memory() {
        cudaCheck(cudaMalloc(&(dev_buf_[input_idx_]), sizeof(float) * batch_sz_ * input_sz_));
        cudaCheck(cudaMalloc(&(dev_buf_[output_idx_]), sizeof(float) * batch_sz_ * output_sz_));
    }
    
    static void thread_func(abstract_queue<inference_msg*>& request_queue,
                            abstract_queue<inference_msg*>& response_queue,
                            inference_engine* engine) {
        cudaCheck(cudaSetDevice(engine->gpu_id_));
        try {
            while (!engine->stop_) {
                //std::cout << "inference engine " << engine->gpu_id_ << " is requesting input data" << std::endl;
                inference_msg *msg = request_queue.pop();
                engine->infer(msg);
                //std::cout << "inference engine " << engine->gpu_id_ << " is pushing output data" << std::endl;
                response_queue.push(msg);
            }
        } catch(queue_closed) {
        }
        std::cout << "inference engine " << engine->gpu_id_ << " has shut down" << std::endl;
    }
public:
    tensorrt_profiler* profiler() { return profiler_; }
    int input_size() const { return batch_sz_*input_sz_; }
    int output_size() const { return batch_sz_*output_sz_; }
    
    int gpu_id() const { return gpu_id_; }
    time_tracker* get_time_tracker() { return &tm_tracker_; }
    
    inference_msg* new_inferene_message(const bool random_input=false) {
        inference_msg *msg = new inference_msg(batch_sz_*input_sz_, batch_sz_*output_sz_);
        if (random_input)
            msg->random_input();
        return msg;
    }
    void reset() {
        reset_ = true;
    }
    void stop() {
        stop_ = true;
    }
    void join() {
        if (internal_thread_ && internal_thread_->joinable())
            internal_thread_->join();
    }
    
    /**
     * @brief Create an instance of inference engine. MUST be called from the main thread.
     * 
     * We need to call it from the main due to sevearl reasons:
     *    - To make sure we correctly bind to particular GPU
     *    - To make sure that the very first engine, in case of using int8, will calibrate
     *      model and cache results that will then be resused by other threads.
     */
    inference_engine(const int gpu_id, tensorrt_logger& logger, const inference_engine_opts& opts)
        : gpu_id_(gpu_id), logger_(logger), batch_sz_(opts.batch_size_), stop_(false),
          reset_(false), tm_tracker_(opts.num_batches_) {
        using namespace nvcaffeparser1;
        int prev_cuda_device(0);
        cudaCheck(cudaGetDevice(&prev_cuda_device));
        cudaCheck(cudaSetDevice(gpu_id_));
        const DataType data_type = opts.data_type();
        logger.log_info("[main] Creating inference builder");
        IBuilder* builder = createInferBuilder(logger_);
        // Parse the caffe model to populate the network, then set the outputs.
        // For INT8 inference, the input model must be specified with 32-bit weights.
        logger.log_info("[main] Creating network and Caffe parser (model: " + opts.model_file_ + ")");
        INetworkDefinition* network = builder->createNetwork();
        ICaffeParser* caffe_parser = createCaffeParser();
        const IBlobNameToTensor* blob_name_to_tensor = caffe_parser->parse(
            opts.model_file_.c_str(), // *.prototxt caffe model definition
            nullptr,       // if null, random weights?
            *network, 
            (data_type == DataType::kINT8 ? DataType::kFLOAT : data_type)
        );
        // Specify what tensors are output tensors.
        network->markOutput(*blob_name_to_tensor->find(opts.output_name_.c_str()));
        // Build the engine.
        builder->setMaxBatchSize(opts.batch_size_);
        builder->setMaxWorkspaceSize(1 << 30); 
        // Half and INT8 precision specific options
        if (data_type == DataType::kHALF) {
            logger.log_info("Enabling FP16 mode");
            builder->setHalf2Mode(true);
        } else if (data_type == DataType::kINT8) {
            logger.log_info("Enabling INT8 mode");
            calibrator_.setBatchSize(opts.batch_size_);

            // Allocate memory but before figure out size of input tensor.
            const nvinfer1::ITensor* input_tensor = blob_name_to_tensor->find(opts.input_name_.c_str());
            calibrator_.initialize(get_tensor_size(input_tensor), 10, opts.model_id_, opts.calibrator_cache_path_);

            builder->setInt8Mode(true);
            builder->setInt8Calibrator(&calibrator_);
        } else {
            logger.log_info("Enabling FP32 mode");
        }
        // This is where we need to use calibrator
        engine_ = builder->buildCudaEngine(*network);
        logger.log_info("[main] Creating execution context");
        exec_ctx_ = engine_->createExecutionContext();
        if (opts.use_profiler_) {
            profiler_ = new tensorrt_profiler();
            exec_ctx_->setProfiler(profiler_);
        }
        logger.log_info("[main] Getting network bindings");
        logger.log_bindings(engine_);
        // We need to figure out number of elements in input/output tensors.
        // Also, we need to figure out their indices.
        const auto num_bindings = engine_->getNbBindings();
        dev_buf_.resize(num_bindings, 0);
        input_idx_ = engine_->getBindingIndex(opts.input_name_.c_str());
        output_idx_ = engine_->getBindingIndex(opts.output_name_.c_str());
        if (input_idx_ < 0) { logger.log_error("Input blob not found."); }
        if (output_idx_ < 0) { logger.log_error("Output blob not found."); }
        input_sz_ = get_binding_size(engine_, input_idx_),    // Number of elements in 'data' tensor.
        output_sz_ = get_binding_size(engine_, output_idx_);  // Number of elements in 'prob' tensor.
            
        // Destroy objects that we do not need anymore
        network->destroy();
        caffe_parser->destroy();
        builder->destroy();
        logger.log_info("[main] Cleaning buffers");
        if (data_type == DataType::kINT8) {
            calibrator_.freeCalibrationMemory();
        }
        cudaCheck(cudaSetDevice(prev_cuda_device));
    }
    
    virtual ~inference_engine() {
        if (profiler_) {
            profiler_->printLayerTimes(nbatches_);
            delete profiler_;
        }
        exec_ctx_->destroy();
        engine_->destroy();
        if (dev_buf_[input_idx_]) cudaFree(dev_buf_[input_idx_]);
        if (dev_buf_[output_idx_]) cudaFree(dev_buf_[output_idx_]);
        if (internal_thread_) delete internal_thread_;
    }
    
    void infer(inference_msg *msg) {
        if (!dev_buf_[input_idx_] || !dev_buf_[output_idx_]) {
            init_device_memory();
        }
        if (reset_) {
            reset_ = false;
            if (profiler_) profiler_->reset();
            tm_tracker_.reset();
        }
        // Copy Input Data to the GPU.
        tm_tracker_.batch_started();
        cudaCheck(cudaMemcpy(dev_buf_[input_idx_], msg->input_.data(), sizeof(float)*msg->input_.size(), cudaMemcpyHostToDevice));
        // Launch an instance of the GIE compute kernel.
        tm_tracker_.infer_started();
        if(!exec_ctx_->execute(batch_sz_, dev_buf_.data())) {logger_.log_error("Kernel was not run");}
        tm_tracker_.infer_done();
        // Copy Output Data to the Host.
        cudaCheck(cudaMemcpy(msg->output_.data(), dev_buf_[output_idx_], sizeof(float)*msg->output_.size(), cudaMemcpyDeviceToHost));
        tm_tracker_.batch_done();
        //
        msg->infer_time_ = tm_tracker_.last_infer_time();
        msg->batch_time_ = tm_tracker_.last_batch_time();
        nbatches_ ++;
    }
    
    void start(abstract_queue<inference_msg*>& request_queue, abstract_queue<inference_msg*>& response_queue) {
        internal_thread_ = new std::thread(
            &inference_engine::thread_func,
            std::ref(request_queue),
            std::ref(response_queue),
            this
        );
    }
};


class mgpu_inference_engine {
    std::vector<inference_engine*> engines_;
    thread_safe_queue<inference_msg*> request_queue_;
    thread_safe_queue<inference_msg*> response_queue_;
public:
    mgpu_inference_engine(std::string gpus, tensorrt_logger& logger, const inference_engine_opts& opts) {
        std::replace(gpus.begin(), gpus.end(), ',', ' ');
        std::istringstream stream(gpus);
        int gpu_id = 0;
        while (stream>>gpu_id) {
            inference_engine *engine = new inference_engine(gpu_id, logger, opts);
            engines_.push_back(engine);
        }
    }
    
    virtual ~mgpu_inference_engine() {
        for (int i=0; i<engines_.size(); ++i) {
            delete engines_[i];
        }
    }
    
    int num_engines() const { return engines_.size(); }
    inference_engine* engine(const int idx) { return engines_[idx]; }
    
    thread_safe_queue<inference_msg*>* request_queue()  { return &request_queue_; }
    thread_safe_queue<inference_msg*>* response_queue() { return &response_queue_; }
    
    int input_size() const { return engines_[0]->input_size(); }
    int output_size() const { return engines_[0]->output_size(); }
    
    bool layer_wise_profiling() { return engines_[0]->profiler() != nullptr; }
    void reset() {
        for(auto& engine : engines_) {
            engine->reset();
        }
    }
    
    void start() {
        for(auto& engine : engines_) {
            engine->start(request_queue_, response_queue_);
        }
    }
    void stop() {
        // Eninges can exist on these both events - whatever happens first.
        request_queue_.close();
        response_queue_.close();
        for(auto& engine : engines_) {
            engine->stop(); 
        } 
    }
    void join() { for(auto& engine : engines_) { engine->join(); } }
};

#endif
