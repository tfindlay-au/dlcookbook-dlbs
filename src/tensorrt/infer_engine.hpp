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

class inference_engine_opts;
std::ostream &operator<<(std::ostream &os, inference_engine_opts const &opts);
void check_bindings(ICudaEngine* engine, const std::string& input_blob, const std::string output_blob, logger_impl& logger);



void check_bindings(ICudaEngine* engine, const std::string& input_blob, const std::string output_blob, logger_impl& logger) {
    const auto nb = engine->getNbBindings();
    if (nb <= 0)
        logger.log_error("Invalid number of a model's IO bindings (" + std::to_string(nb) + ").");
    const auto ib = engine->getBindingIndex(input_blob.c_str());
    const auto ob = engine->getBindingIndex(output_blob.c_str());
    if (ib < 0 || ob < 0 || ib > nb || ob > nb)
        logger.log_error(
            "Invalid indice(s) of IO blob(s). Number of bindings = " + std::to_string(nb) +
            ", input blob (" + input_blob + ") index = " + std::to_string(ib) +
            ", output blob (" + output_blob + ") index = " + std::to_string(ob)
        );
}

/**
 * @brief Options to instantiate an inference engine.
 */
struct inference_engine_opts {
    std::string model_file_;                   //!< Full path to a Caffe's protobuf inference descriptor.
    std::string dtype_ = "float32";            //!< Data type (precision): float32(float), float16, int8.
    size_t batch_size_ = 16;                   //!< Batch size.
    size_t num_warmup_batches_ = 10;           //!< Number of warmup batches.
    size_t num_batches_ = 100;                 //!< Number of benchmark batches.
    bool use_profiler_ = false;                //!< Perform layer-wise model profiling.
    std::string input_name_ = "data";          //!< Name of an input blob.
    std::string output_name_ = "prob";         //!< Name of an output blob.
    
    std::string model_id_;                     //!< Model ID, used to store calibrator cache if data type is int8.
    std::string calibrator_cache_path_ = "";   //!< Path to store calibrator cache if data type is int8.
    
    std::vector<int> gpus_;                    //!< GPUs to use.
    size_t report_frequency_ = 0;              //!< If > 0, report intermidiate progress every this number of batches time GPUs count.
    size_t inference_queue_size_ = 4;          //!< Size of input request queue.
    bool do_not_report_batch_times_ = false;   //!< If true, do not log per-batch performance.
    
    bool fake_inference_ = false;              //!< If true, perform fake inference.
    
    DataType data_type() const {
        if (dtype_ == "float32" || dtype_ == "float")
            return DataType::kFLOAT;
        if (dtype_ == "float16")
            return DataType::kHALF;
        return DataType::kINT8;
    }
};
std::ostream &operator<<(std::ostream &os, inference_engine_opts const &opts) {
    os << "[inference_engine_opts ]: " 
       << "model_file=" << opts.model_file_ << ", dtype=" << opts.dtype_ << ", batch_size=" << opts.batch_size_
       << ", num_warmup_batches=" << opts.num_warmup_batches_ << ", num_batches=" << opts.num_batches_
       << ", use_profiler=" << opts.use_profiler_ << ", input_blob_name=" << opts.input_name_
       << ", output_blob_name=" << opts.output_name_ << ", model_id=" << opts.model_id_
       << ", calibrator_cache_path=" << opts.calibrator_cache_path_
       << ", fake inference=" << (opts.fake_inference_ ? "true" : "false");
    return os;
}

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
protected:
    int engine_id_;           //!< Engine ID, same as GPU ID. Negative ID identifies fake inference engine.
    int num_engines_;         //!< Total number of engines if managed by mGPU inference engine

    logger_impl& logger_;
    time_tracker tm_tracker_;
    
    size_t nbatches_ = 0;
    
    size_t batch_sz_ = 0;
    size_t input_sz_ = 0;
    size_t output_sz_ = 0;
    
    std::atomic_bool stop_;
    std::atomic_bool reset_;
 
    std::thread *internal_thread_ = nullptr;
private:
    static void thread_func(abstract_queue<inference_msg*>& request_queue,
                            abstract_queue<inference_msg*>& response_queue,
                            inference_engine* engine) {
        if (engine->engine_id_ >= 0) {
            cudaCheck(cudaSetDevice(engine->engine_id_));
        }
        running_average fetch, process, submit;
        try {
            timer queue_timer;
            while (!engine->stop_) {
                queue_timer.restart();  inference_msg *msg = request_queue.pop();  fetch.update(queue_timer.ms_elapsed());
                queue_timer.restart();  engine->infer(msg);                        process.update(queue_timer.ms_elapsed());
                queue_timer.restart();  response_queue.push(msg);                  submit.update(queue_timer.ms_elapsed());
            }
        } catch(queue_closed) {
        }
        engine->logger_.log_info(fmt(
            "[inference engine %02d/%02d]: {fetch:%.5f}-->--[process:%.5f]-->--{submit:%.5f}",
            abs(engine->engine_id_), engine->num_engines_, fetch.value(), process.value(), submit.value()
        ));
    }
public:
    size_t batch_size() const { return batch_sz_; }
    size_t input_size() const { return batch_sz_*input_sz_; }
    size_t output_size() const { return batch_sz_*output_sz_; }
    
    int engine_id() const { return engine_id_; }
    time_tracker* get_time_tracker() { return &tm_tracker_; }
    
    inference_msg* new_inferene_message(const bool random_input=false) {
        inference_msg *msg = new inference_msg(batch_sz_, batch_sz_*input_sz_, batch_sz_*output_sz_);
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
    inference_engine(const int engine_id, const int num_engines, logger_impl& logger, const inference_engine_opts& opts)
        : engine_id_(engine_id), num_engines_(num_engines), logger_(logger), tm_tracker_(opts.num_batches_),
          batch_sz_(opts.batch_size_), input_sz_(3 * 227 * 227), output_sz_(1000),
          stop_(false), reset_(false) {}
    
    virtual ~inference_engine() {
        if (internal_thread_) delete internal_thread_;
    }
    
    void start(abstract_queue<inference_msg*>& request_queue, abstract_queue<inference_msg*>& response_queue) {
        internal_thread_ = new std::thread(
            &inference_engine::thread_func,
            std::ref(request_queue),
            std::ref(response_queue),
            this
        );
    }
    
    virtual void infer(inference_msg *msg) = 0;
};

class fake_inference_engine : public inference_engine {
public:
    fake_inference_engine(const int engine_id, const int num_engines,
                          logger_impl & logger, const inference_engine_opts& opts)
        : inference_engine(-1*(engine_id+1), num_engines_, logger, opts) {}
    void infer(inference_msg *msg) override {};
};

class tensorrt_inference_engine : public inference_engine {
private:
    calibrator_impl calibrator_;
    profiler_impl *profiler_ = nullptr;
    
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* exec_ctx_ = nullptr;
    
    std::vector<void*> dev_buf_;
    size_t input_idx_ = 0;
    size_t output_idx_ = 0;
private:
    void init_device_memory() {
        cudaCheck(cudaMalloc(&(dev_buf_[input_idx_]), sizeof(float) * batch_sz_ * input_sz_));
        cudaCheck(cudaMalloc(&(dev_buf_[output_idx_]), sizeof(float) * batch_sz_ * output_sz_));
    }
public:
    profiler_impl* profiler() { return profiler_; }
    /**
     * @brief Create an instance of inference engine. MUST be called from the main thread.
     * 
     * We need to call it from the main due to sevearl reasons:
     *    - To make sure we correctly bind to particular GPU
     *    - To make sure that the very first engine, in case of using int8, will calibrate
     *      model and cache results that will then be resused by other threads.
     */
    tensorrt_inference_engine(const int engine_id, const int num_engines,
                              logger_impl& logger, const inference_engine_opts& opts)
        : inference_engine(engine_id, num_engines, logger, opts), calibrator_(logger) {
        using namespace nvcaffeparser1;
        const auto me = fmt("[inference engine %02d/%02d]:", engine_id, num_engines);
        int prev_cuda_device(0);
        cudaCheck(cudaGetDevice(&prev_cuda_device));
        cudaCheck(cudaSetDevice(engine_id));
        const DataType data_type = opts.data_type();
        logger.log_info(me + " Creating inference builder");
        IBuilder* builder = createInferBuilder(logger_);
        // Parse the caffe model to populate the network, then set the outputs.
        // For INT8 inference, the input model must be specified with 32-bit weights.
        logger.log_info(me + " Creating network and Caffe parser (model: " + opts.model_file_ + ")");
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
                             
            logger.log_info(me + " Enabling FP16 mode");
            builder->setHalf2Mode(true);
        } else if (data_type == DataType::kINT8) {
            logger.log_info(me + " Enabling INT8 mode");
            calibrator_.setBatchSize(opts.batch_size_);

            // Allocate memory but before figure out size of input tensor.
            const nvinfer1::ITensor* input_tensor = blob_name_to_tensor->find(opts.input_name_.c_str());
            calibrator_.initialize(get_tensor_size(input_tensor), 10, opts.model_id_, opts.calibrator_cache_path_);

            builder->setInt8Mode(true);
            builder->setInt8Calibrator(&calibrator_);
        } else {
            logger.log_info(me + " Enabling FP32 mode");
        }
        // This is where we need to use calibrator
        engine_ = builder->buildCudaEngine(*network);
        logger.log_info(me + " Creating execution context");
        exec_ctx_ = engine_->createExecutionContext();
        if (opts.use_profiler_) {
            profiler_ = new profiler_impl();
            exec_ctx_->setProfiler(profiler_);
        }
        logger.log_info(me + " Getting network bindings");
        logger.log_bindings(engine_);
        // We need to figure out number of elements in input/output tensors.
        // Also, we need to figure out their indices.
        check_bindings(engine_, opts.input_name_, opts.output_name_, logger_);
        dev_buf_.resize(static_cast<size_t>(engine_->getNbBindings()), 0);
        input_idx_ = static_cast<size_t>(engine_->getBindingIndex(opts.input_name_.c_str()));
        output_idx_ = static_cast<size_t>(engine_->getBindingIndex(opts.output_name_.c_str()));
        input_sz_ = get_binding_size(engine_, input_idx_),    // Number of elements in 'data' tensor.
        output_sz_ = get_binding_size(engine_, output_idx_);  // Number of elements in 'prob' tensor.
        // Destroy objects that we do not need anymore
        network->destroy();
        caffe_parser->destroy();
        builder->destroy();
        logger.log_info(me + " Cleaning buffers");
        if (data_type == DataType::kINT8) {
            calibrator_.freeCalibrationMemory();
        }
        cudaCheck(cudaSetDevice(prev_cuda_device));
    }
    
    ~tensorrt_inference_engine() {
        if (profiler_) {
            profiler_->printLayerTimes(nbatches_);
            delete profiler_;
        }
        exec_ctx_->destroy();
        engine_->destroy();
        if (dev_buf_[input_idx_]) cudaFree(dev_buf_[input_idx_]);
        if (dev_buf_[output_idx_]) cudaFree(dev_buf_[output_idx_]);
    }
    
    void infer(inference_msg *msg) override {
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
        cudaCheck(cudaMemcpy(dev_buf_[input_idx_], msg->input().data(), sizeof(float)*msg->input().size(), cudaMemcpyHostToDevice));
        // Launch an instance of the GIE compute kernel.
        tm_tracker_.infer_started();
        if(!exec_ctx_->execute(batch_sz_, dev_buf_.data())) {logger_.log_error("Kernel was not run");}
        tm_tracker_.infer_done();
        // Copy Output Data to the Host.
        cudaCheck(cudaMemcpy(msg->output().data(), dev_buf_[output_idx_], sizeof(float)*msg->output().size(), cudaMemcpyDeviceToHost));
        tm_tracker_.batch_done();
        //
        msg->set_infer_time(tm_tracker_.last_infer_time());
        msg->set_batch_time(tm_tracker_.last_batch_time());
        nbatches_ ++;
    }
};


class mgpu_inference_engine {
    std::vector<inference_engine*> engines_;
    thread_safe_queue<inference_msg*> request_queue_;
    thread_safe_queue<inference_msg*> response_queue_;
public:
    mgpu_inference_engine(const inference_engine_opts& opts, logger_impl& logger) {
        const int num_engines = static_cast<int>(opts.gpus_.size());
        for (size_t i=0; i<opts.gpus_.size(); ++i) {
            inference_engine *engine = nullptr;
            const auto gpu_id = opts.gpus_[i];
            if (opts.fake_inference_) {
                engine = new fake_inference_engine(static_cast<int>(gpu_id), num_engines, logger, opts);
            } else {
                engine = new tensorrt_inference_engine(static_cast<int>(gpu_id), num_engines, logger, opts);
            }
            engines_.push_back(engine);
        }
    }
    
    virtual ~mgpu_inference_engine() {
        for (size_t i=0; i<engines_.size(); ++i) {
            delete engines_[i];
        }
    }
    
    size_t num_engines() const { return engines_.size(); }
    inference_engine* engine(const size_t idx) { return engines_[idx]; }
    
    thread_safe_queue<inference_msg*>* request_queue()  { return &request_queue_; }
    thread_safe_queue<inference_msg*>* response_queue() { return &response_queue_; }
    
    size_t batch_size() const { return engines_[0]->batch_size(); }
    size_t input_size() const { return engines_[0]->input_size(); }
    size_t output_size() const { return engines_[0]->output_size(); }
    
    //bool layer_wise_profiling() { return engines_[0]->profiler() != nullptr; }
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
