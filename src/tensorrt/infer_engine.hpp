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
#include "infer_task.hpp"
#include "calibrator.hpp"
#include <NvCaffeParser.h>


class infer_engine {
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
    
    static void thread_func(abstract_queue<infer_task*>& input_queue,
                            abstract_queue<infer_task*>& output_queue,
                            infer_engine* engine) {
        cudaCheck(cudaSetDevice(engine->gpu_id_));
        infer_task* task(nullptr);
        while (!engine->stop_ && (task=input_queue.pop()) != nullptr) {
            engine->infer(task);
            output_queue.push(task);
        }
    }
public:
    tensorrt_profiler* profiler() { return profiler_; }
    int input_size() const { return batch_sz_*input_sz_; }
    int output_size() const { return batch_sz_*output_sz_; }
    
    int gpu_id() const { return gpu_id_; }
    time_tracker* get_time_tracker() { return &tm_tracker_; }
    
    infer_task* new_task(const bool random_input=false) {
        infer_task* task = new infer_task(batch_sz_*input_sz_, batch_sz_*output_sz_);
        if (random_input)
            task->random_input();
        return task;
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
    infer_engine(const int gpu_id,
                 tensorrt_logger& logger,
                 const std::string& model_id,
                 const std::string& model_file,
                 std::string dtype,
                 const int batch_size,
                 const int num_benchmark_batches,
                 const bool use_profiler=false,
                 const std::string& output_name="prob",
                 const std::string& input_name="data",
                 const std::string& calibrator_cache_path=""
                ) : gpu_id_(gpu_id), logger_(logger), batch_sz_(batch_size), stop_(false),
                    reset_(false), tm_tracker_(num_benchmark_batches) {
        using namespace nvcaffeparser1;
        int prev_cuda_device(0);
        cudaCheck(cudaGetDevice(&prev_cuda_device));
        cudaCheck(cudaSetDevice(gpu_id_));
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
        IBuilder* builder = createInferBuilder(logger_);
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
            calibrator_.setBatchSize(batch_size);

            // Allocate memory but before figure out size of input tensor.
            const nvinfer1::ITensor* input_tensor = blob_name_to_tensor->find(input_name.c_str());
            calibrator_.initialize(get_tensor_size(input_tensor), 10, model_id, calibrator_cache_path);

            builder->setInt8Mode(true);
            builder->setInt8Calibrator(&calibrator_);
        } else {
            logger.log_info("Enabling FP32 mode");
        }
        // This is where we need to use calibrator
        engine_ = builder->buildCudaEngine(*network);
        logger.log_info("[main] Creating execution context");
        exec_ctx_ = engine_->createExecutionContext();
        if (use_profiler) {
            profiler_ = new tensorrt_profiler();
            exec_ctx_->setProfiler(profiler_);
        }
        logger.log_info("[main] Getting network bindings");
        logger.log_bindings(engine_);
        // We need to figure out number of elements in input/output tensors.
        // Also, we need to figure out their indices.
        const auto num_bindings = engine_->getNbBindings();
        dev_buf_.resize(num_bindings, 0);
        input_idx_ = engine_->getBindingIndex(input_name.c_str());
        output_idx_ = engine_->getBindingIndex(output_name.c_str());
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
    
    virtual ~infer_engine() {
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
    
    void infer(infer_task* task) {
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
        cudaCheck(cudaMemcpy(dev_buf_[input_idx_], task->input_.data(), sizeof(float)*task->input_.size(), cudaMemcpyHostToDevice));
        // Launch an instance of the GIE compute kernel.
        tm_tracker_.infer_started();
        if(!exec_ctx_->execute(batch_sz_, dev_buf_.data())) {logger_.log_error("Kernel was not run");}
        tm_tracker_.infer_done();
        // Copy Output Data to the Host.
        cudaCheck(cudaMemcpy(task->output_.data(), dev_buf_[output_idx_], sizeof(float)*task->output_.size(), cudaMemcpyDeviceToHost));
        tm_tracker_.batch_done();
        //
        task->infer_time_ = tm_tracker_.last_infer_time();
        task->batch_time_ = tm_tracker_.last_batch_time();
        nbatches_ ++;
    }
    
    void start(abstract_queue<infer_task*>& input_queue, abstract_queue<infer_task*>& output_queue) {
        internal_thread_ = new std::thread(&infer_engine::thread_func, std::ref(input_queue), std::ref(output_queue), this);
    }
};


class infer_engine_pool {
    std::vector<infer_engine*> engines_;
    abstract_queue<infer_task*>* data_queue_;
public:
    infer_engine_pool(std::string gpus, tensorrt_logger& logger, const std::string& model_id,
                      const std::string& model_file, std::string dtype, const int batch_size,
                      const int num_benchmark_batches, const bool use_profiler=false, const std::string& output_name="prob",
                      const std::string& input_name="data", const std::string& calibrator_cache_path="") {
        // IMPLEMENT mE!!!!!
        std::replace(gpus.begin(), gpus.end(), ',', ' ');
        std::istringstream stream(gpus);
        int gpu_id = 0;
        while (stream>>gpu_id) {
            infer_engine *engine = new infer_engine(gpu_id, logger, model_id, model_file, dtype,
                                                    batch_size, num_benchmark_batches, use_profiler,
                                                    output_name, input_name, calibrator_cache_path);
            engines_.push_back(engine);
        }
    }
    
    virtual ~infer_engine_pool() {
        for (int i=0; i<engines_.size(); ++i) {
            delete engines_[i];
        }
    }
    
    int num_engines() const { return engines_.size(); }
    infer_engine* engine(const int idx) { return engines_[idx]; }
    
    int input_size() const { return engines_[0]->input_size(); }
    int output_size() const { return engines_[0]->output_size(); }
    
    bool layer_wise_profiling() { return engines_[0]->profiler() != nullptr; }
    void reset() {
        for(auto& engine : engines_) {
            engine->reset();
        }
    }
    
    void start(abstract_queue<infer_task*>& data_queue, abstract_queue<infer_task*>& output_queue) {
        data_queue_ = &data_queue;
        for(auto& engine : engines_) {
            engine->start(data_queue, output_queue);
        }
    }
    void stop() { 
        for(auto& engine : engines_) {
            data_queue_->push(nullptr);
            engine->stop(); 
        } 
    }
    void join() { for(auto& engine : engines_) { engine->join(); } }
};

#endif
