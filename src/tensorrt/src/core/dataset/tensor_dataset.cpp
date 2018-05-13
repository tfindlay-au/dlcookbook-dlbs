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

#include "core/dataset/tensor_dataset.hpp"

void tensor_dataset::prefetcher_func(tensor_dataset* myself,
                                     const size_t prefetcher_id, const size_t num_prefetchers) {
    sharded_vector<std::string> my_files(myself->file_names_, myself->prefetchers_.size(), prefetcher_id);
        
    const int height(static_cast<int>(myself->opts_.height_)),
              width(static_cast<int>(myself->opts_.width_));
    const size_t img_size = 3 * height * width;
    running_average fetch, load, submit;
    inference_msg *output(nullptr);
    try {
        timer clock;
        clock.restart();
        while (!myself->stop_) {
            // Get inference request
            clock.restart();
            output = myself->inference_msg_pool_->get();
            fetch.update(clock.ms_elapsed());
            for (size_t i=0; i<output->batch_size(); ++i) {
                const auto fname = my_files.next();
                int img_nchannels(0), img_width(0), img_height(0);
                    
                std::ifstream in(fname.c_str());
                in.read((char*)&img_nchannels, sizeof(int));
                in.read((char*)&img_width, sizeof(int));
                in.read((char*)&img_height, sizeof(int));
                if (img_nchannels != 3 || img_width != width || img_height != height) {
                    myself->logger_.log_error(fmt(
                        "[prefetcher       %02d/%02d]: invalid image dimensions, epxecting (3, %d, %d), received (3, %d, %d)",
                        prefetcher_id, num_prefetchers, width, height, img_width, img_height    
                    ));
                }
                in.read((char*)(output->input().data()+img_size*i), sizeof(float)*img_size);
            }
            // Submit inference request
            clock.restart();  myself->request_queue_->push(output);  submit.update(clock.ms_elapsed());
        }
    } catch(queue_closed) {
    }
    myself->logger_.log_info(fmt(
        "[prefetcher       %02d/%02d]: {fetch:%.5f}-->--[load:%.5f]-->--{submit:%.5f}",
        prefetcher_id, num_prefetchers, fetch.value(), load.value(), submit.value()
    ));
}

tensor_dataset::tensor_dataset(const dataset_opts& opts, inference_msg_pool* pool,
                               abstract_queue<inference_msg*>* request_queue, logger_impl& logger)
: dataset(pool, request_queue), opts_(opts), logger_(logger) {

    fs_utils::initialize_dataset(opts_.data_dir_, file_names_);
    prefetchers_.resize(opts_.num_prefetchers_, nullptr);
}
    
void tensor_dataset::run() {
    // Run prefetch workers
    for (size_t i=0; i<prefetchers_.size(); ++i) {
        prefetchers_[i] = new std::thread(&(tensor_dataset::prefetcher_func), this, i, prefetchers_.size());
    }
    // Wait
    for (auto& prefetcher : prefetchers_) {
        if (prefetcher->joinable()) prefetcher->join();
        delete prefetcher;
    }
}

float tensor_dataset::benchmark(const std::string dir, const size_t batch_size, const size_t img_size,
                                const size_t num_prefetches, const size_t num_infer_msgs,
                                const int num_warmup_batches, const int num_batches) {
    logger_impl logger;
    dataset_opts opts;
    opts.data_dir_ = dir;
    opts.num_prefetchers_ = num_prefetches;
    opts.prefetch_batch_size_=batch_size;
    opts.height_ = img_size;
    opts.width_ = img_size;
        
    inference_msg_pool pool(num_infer_msgs, opts.prefetch_batch_size_, 3*opts.height_*opts.width_, 1000);
    thread_safe_queue<inference_msg*> request_queue;
    tensor_dataset data(opts, &pool, &request_queue, logger);
        
    data.start();
    // N warmup iterations
    std::cout << "Running warmup iterations" << std::endl;
    for (int i=0; i<num_warmup_batches; ++i) {
        pool.release(request_queue.pop());
    }
    // N benchmark iterations
    std::cout << "Running benchmark iterations" << std::endl;
    timer t;
    size_t num_images(0);
    for (int i=0; i<num_batches; ++i) {
        inference_msg* msg = request_queue.pop();
        num_images += msg->batch_size();
        pool.release(msg);
    }
    const float throughput = 1000.0 * num_images / t.ms_elapsed();
    data.stop();
    std::cout << "num_readers=" << opts.num_prefetchers_ << ", throughput=" << throughput << std::endl;
    data.join();
    return throughput;
}
