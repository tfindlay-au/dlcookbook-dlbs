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

#ifndef DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_ENGINE
#define DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_ENGINE

#include "core/infer_engine.hpp"
#include "engines/tensorrt/calibrator.hpp"
#include "engines/tensorrt/profiler.hpp"

#include <NvCaffeParser.h>

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
    void init_device_memory();
public:
    profiler_impl* profiler() { return profiler_; }
    tensorrt_inference_engine(const int engine_id, const int num_engines,
                              logger_impl& logger, const inference_engine_opts& opts);
    ~tensorrt_inference_engine();
    void init_device() override;
    void infer(inference_msg *msg) override;
};

#endif
