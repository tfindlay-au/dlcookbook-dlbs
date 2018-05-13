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
#include "core/dataset/dataset.hpp"


std::ostream &operator<<(std::ostream &os, dataset_opts const &opts) {
    os << "[dataset_opts       ] "
       << "data_dir=" << opts.data_dir_ << ", resize_method=" << opts.resize_method_ << ", height=" << opts.height_
       << ", width=" << opts.width_ << ", num_prefetchers=" << opts.num_prefetchers_
       << ", num_decoders=" << opts.num_decoders_ << ", prefetch_batch_size=" << opts.prefetch_batch_size_
       << ", prefetch_queue_size=" << opts.prefetch_queue_size_
       << ", fake_decoder=" << (opts.fake_decoder_ ? "true" : "false")
       << ", data_name=" << opts.data_name_;
    return os;
}

void synthetic_dataset::run() {
    try {
        while(!stop_) {
            inference_msg *msg = inference_msg_pool_->get();
            request_queue_->push(msg);
        }
    } catch(queue_closed) {
    }
}