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

#include "engines/tensorrt_engine.hpp"

DataType str2dtype(const std::string& dtype) {
    if (dtype == "float32" || dtype == "float")
        return DataType::kFLOAT;
    if (dtype == "float16")
        return DataType::kHALF;
    return DataType::kINT8;
}

void tensorrt_inference_engine::init_device_memory() {
    cudaCheck(cudaMalloc(&(dev_buf_[input_idx_]), sizeof(float) * batch_sz_ * input_sz_));
    cudaCheck(cudaMalloc(&(dev_buf_[output_idx_]), sizeof(float) * batch_sz_ * output_sz_));
}

tensorrt_inference_engine::tensorrt_inference_engine(const int engine_id, const int num_engines,
                                                     logger_impl& logger, const inference_engine_opts& opts)
: inference_engine(engine_id, num_engines, logger, opts), calibrator_(logger) {

    using namespace nvcaffeparser1;
    const auto me = fmt("[inference engine %02d/%02d]:", engine_id, num_engines);
    int prev_cuda_device(0);
    cudaCheck(cudaGetDevice(&prev_cuda_device));
    cudaCheck(cudaSetDevice(engine_id));
    const DataType data_type = str2dtype(opts.dtype_);
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
    
tensorrt_inference_engine::~tensorrt_inference_engine() {
    if (profiler_) {
        profiler_->printLayerTimes(nbatches_);
        delete profiler_;
    }
    exec_ctx_->destroy();
    engine_->destroy();
    if (dev_buf_[input_idx_]) cudaFree(dev_buf_[input_idx_]);
    if (dev_buf_[output_idx_]) cudaFree(dev_buf_[output_idx_]);
}
    
void tensorrt_inference_engine::init_device() {
    cudaCheck(cudaSetDevice(engine_id_));
}
    
void tensorrt_inference_engine::infer(inference_msg *msg) {
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
    cudaCheck(cudaMemcpy(
        dev_buf_[input_idx_],
        msg->input(),
        sizeof(float)*msg->input_size()*msg->batch_size(),
        cudaMemcpyHostToDevice
    ));
    // Launch an instance of the GIE compute kernel.
    tm_tracker_.infer_started();
    if(!exec_ctx_->execute(batch_sz_, dev_buf_.data())) {logger_.log_error("Kernel was not run");}
    tm_tracker_.infer_done();
    // Copy Output Data to the Host.
    cudaCheck(cudaMemcpy(
        msg->output(),
        dev_buf_[output_idx_],
        sizeof(float)*msg->output_size()*msg->batch_size(),
        cudaMemcpyDeviceToHost
    ));
    tm_tracker_.batch_done();
    //
    msg->set_infer_time(tm_tracker_.last_infer_time());
    msg->set_batch_time(tm_tracker_.last_batch_time());
    nbatches_ ++;
}
