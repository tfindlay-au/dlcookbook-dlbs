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
// https://juanchopanzacpp.wordpress.com/2013/02/26/concurrent-queue-c11/
// https://codereview.stackexchange.com/questions/149676/writing-a-thread-safe-queue-in-c
#ifndef DLBS_TENSORRT_BACKEND_DATA_PROVIDERS
#define DLBS_TENSORRT_BACKEND_DATA_PROVIDERS

#include "infer_msg.hpp"
#include "logger.hpp"

class image_provider_opts;
std::ostream &operator<<(std::ostream &os, image_provider_opts const &opts);

/**
 * @brief Base abstract class for all data providers.
 */
class data_provider {
private:
    //!< A data provider can be 'started'. In this case, this is the thread object.
    std::thread *internal_thread_;
    //!< A function that's invoked when internal thread is started. The only
    //!< purpose of this function is to call @see run method that should implement logic.
    static void thread_func(data_provider* provider) {
        provider->run();
    }
protected:
    inference_msg_pool* inference_msg_pool_;         //!< [input]  A pool of free tasks that can be reused to submit infer requests.
    abstract_queue<inference_msg*>* request_queue_;  //!< [output] An output data queue containing requests with real data.
    std::atomic_bool stop_;                          //!< The 'stop' flag indicating internal thread must stop.
public:
    data_provider(inference_msg_pool* pool, abstract_queue<inference_msg*>* request_queue) 
        : inference_msg_pool_(pool), request_queue_(request_queue), stop_(false) {
    }
    virtual ~data_provider() {
        if (internal_thread_) delete internal_thread_;
    }
    //!< Starts internal thread to load data into the data queue.
    void start() {
        internal_thread_ = new std::thread(&data_provider::thread_func, this);
    }
    //!< Requests to stop internal thread and returns without waiting.
    virtual void stop() {
        stop_ = true;
    }
    //!< Waits for internal thread to shutdown.
    void join() {
        if (internal_thread_ && internal_thread_->joinable())
            internal_thread_->join();
    }
    /**
     * @brief Worker function called from internal thread.
     * 
     * It can create any other number of threads. This function needs to
     * fetch a task structure from task pool, fill it with data and pass it
     * to data queue.
     */
    virtual void run() = 0;
};

/**
 * @brief A very simple implementation of a data provider that just fetches tasks
 * from task pool and passes it immidiately to data queue.
 * Can be used to run inference benchmarks with synthetic data. The data in tasks
 * is randomly initialized and has correct shape for requested neural network.
 */
class synthetic_data_provider : public data_provider {
private:
    using data_provider::inference_msg_pool_;
    using data_provider::request_queue_;
    using data_provider::stop_;
public:
    synthetic_data_provider(inference_msg_pool* pool, abstract_queue<inference_msg*>* request_queue) 
        : data_provider(pool, request_queue) {
    }
    void run() override {
        try {
            while(true) {
                if (stop_) break;
                inference_msg *msg = inference_msg_pool_->get();
                if (stop_) break;
                request_queue_->push(msg);
            }
        } catch(queue_closed) {
        }
    }
};

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>

/**
 * @brief Resize method.
 * If an image already has requried shape, no operation is performed. If 'crop'
 * is selected, and an image has smaller size, 'resize' is used instead.
 * Crop is basically a free operation (I think OpenCV just updates matrix header),
 * resize is expensive.
 * Resizing is done on CPUs in one thread. There can be multiple decoders though 
 * decoding different batches in parallel.
 */
enum class resize_method : int {
    crop = 1,    //!< Crop images, if an image has smaller size, resize instead.
    resize = 2   //!< Always resize.
};

struct image_provider_opts {
    std::string data_dir_;                 //!< Path to a dataset.

    std::string resize_method_ = "crop";   //!< Image resize method - 'crop' or 'resize'.
    
    size_t num_prefetchers_ = 0;           //!< Number of prefetch threads (data readers).
    size_t num_decoders_ = 0;              //!< Number of decoder threads (OpenCV -> std::vector conversion). 
    
    size_t prefetch_batch_size_ = 0;       //!< Same as neural network batch size.
    size_t prefetch_queue_size_ = 0;       //!< Maximal size of prefetched batches.
    
    bool fake_decoder_ = false;            //!< If true, decoder does not decode images but passes inference requests through itself.
    
    size_t height_ = 0;                    //!< This is the target image height. Depends on neural network input.
    size_t width_ = 0;                     //!< This is the target image width. Depends on neural network input.
    
    resize_method get_resize_method() const {
        return resize_method_ == "resize"? resize_method::resize : resize_method::crop;
    }
};
std::ostream &operator<<(std::ostream &os, image_provider_opts const &opts) {
    os << "[image_provider_opts       ] "
       << "data_dir=" << opts.data_dir_ << ", resize_method=" << opts.resize_method_ << ", height=" << opts.height_
       << ", width=" << opts.width_ << ", num_prefetchers=" << opts.num_prefetchers_
       << ", num_decoders=" << opts.num_decoders_ << ", prefetch_batch_size=" << opts.prefetch_batch_size_
       << ", prefetch_queue_size=" << opts.prefetch_queue_size_
       << ", fake_decoder=" << (opts.fake_decoder_ ? "true" : "false");
    return os;
}

/**
 * @brief Sharded vector iterates forever over provided chunk. For instance, we can have
 * a vector of file names of images. We then can have multiple image readers that will read
 * images from their own chunk.
 */
template <typename T>
class sharded_vector {
private:
    std::vector<T>* vec_;
    size_t first_idx_;
    size_t length_;
    size_t pos_;
public:
    sharded_vector(std::vector<T>& vec, const size_t first_idx, const size_t length)
    : vec_(&vec), first_idx_(first_idx), length_(length), pos_(first_idx_) {
        if (first_idx_ + length_ >= vec_->size()) {
            length_ = vec_->size() - first_idx;
        }
    }
    
    T& next() {
        T& item = (*vec_)[pos_++];
        if (pos_ >= length_)
            pos_ = first_idx_;
        return item;
    }
};

/**
 * @brief A message containing images read from some storage. An image reader
 * batches images before sending them. Usually, there can be multiple image
 * readers working in parallel threads.
 */
struct prefetch_msg {
    std::vector<cv::Mat> images_;
    size_t num_images() const { return images_.size(); }
};

class image_provider : public data_provider {
private:
    using data_provider::inference_msg_pool_;
    using data_provider::request_queue_;
    using data_provider::stop_;
    
    std::vector<std::string> file_names_;
    std::vector<std::thread*> prefetchers_;
    std::vector<std::thread*> decoders_;
    
    thread_safe_queue<prefetch_msg*> prefetch_queue_;
    image_provider_opts opts_;
    logger_impl& logger_;
private:
    static void prefetcher_func(image_provider* myself, 
                                const size_t prefetcher_id, const size_t num_prefetchers) {
        // Find out images I am responsible for.
        size_t shard_pos(0), shard_length(0);
        get_my_shard(myself->file_names_.size(), myself->prefetchers_.size(),
                     prefetcher_id, shard_pos, shard_length);
        sharded_vector<std::string> my_files(myself->file_names_, shard_pos, shard_length);
        prefetch_msg *msg = new prefetch_msg();
        running_average load, submit;
        try {
            timer clock;
            clock.restart();
            while (!myself->stop_) {
                const auto fname = my_files.next();
                cv::Mat img = cv::imread(fname);
                if (img.data == nullptr) {
                    myself->logger_.log_warning("Error loading image from file: " + fname);
                    continue;
                }
                msg->images_.push_back(img);
                if (msg->num_images() >= myself->opts_.prefetch_batch_size_) {
                    load.update(clock.ms_elapsed());
                    clock.restart();  myself->prefetch_queue_.push(msg);  submit.update(clock.ms_elapsed());
                    msg = new prefetch_msg();
                }
            }
        } catch(queue_closed) {
        }
        myself->logger_.log_info(fmt(
            "[prefetcher       %02d/%02d]: [load:%.5f]-->--{submit:%.5f}",
            prefetcher_id, num_prefetchers, load.value(), submit.value()
        ));
        delete msg;
    }
    
    static void decoder_func(image_provider* myself, const int decoder_id, const int num_decoders) {
        const int height(static_cast<int>(myself->opts_.height_)),
                  width(static_cast<int>(myself->opts_.width_));
        const resize_method resizem = myself->opts_.get_resize_method();
        const size_t image_size = static_cast<size_t>(3 * height * width);
        running_average fetch_imgs, fetch_reqs, process, submit;
        try {
            timer clock;
            inference_msg *output(nullptr);      // Current inference request.
            prefetch_msg *input(nullptr  );      // Current non-decoded images.
            size_t input_cursor(0),              // Current position in input data.
                   output_cursor(0);             // Current position in output data.
            float decode_time(0);
                              
            while(!myself->stop_) {
                // Get free task from the task pool
                if (!output) {
                    clock.restart();
                    output = myself->inference_msg_pool_->get();
                    fetch_reqs.update(clock.ms_elapsed());
                    output_cursor = 0;
                }
                // Get prefetched images
                if (!input) {
                    clock.restart();
                    input = myself->prefetch_queue_.pop();
                    fetch_imgs.update(clock.ms_elapsed());
                    input_cursor = 0;
                }
                // If output messages is filled with data, send it
                const auto need_to_decode = output->batch_size() - output_cursor;
                if (need_to_decode == 0) {
                    process.update(decode_time);
                    clock.restart();
                    myself->request_queue_->push(output);
                    submit.update(clock.ms_elapsed());
                    output = nullptr;
                    decode_time = 0;
                    continue;
                }
                // If there's no data that needs to be decoded, get it.
                const auto can_decode = input->num_images() - input_cursor;
                if (can_decode == 0) {
                    delete input;
                    input = nullptr;
                    continue;
                }
                // This number of instances I will decode
                const auto will_decode = std::min(need_to_decode, can_decode);
                clock.restart();
                if (!myself->opts_.fake_decoder_) {
                    for (size_t i=0; i<will_decode; ++i) {
                        cv::Mat img = input->images_[input_cursor];
                        if (img.rows != height || img.cols != width) {
                            if (resizem == resize_method::resize || img.rows < height || img.cols < width) {
                                cv::resize(input->images_[input_cursor], img, cv::Size(height, width), 0, 0, cv::INTER_LINEAR);
                            } else {
                                img = img(cv::Rect(0, 0, height, width));
                            }
                        }
                        std::copy(
                            img.begin<float>(),
                            img.end<float>(),
                            output->input().begin() + static_cast<std::vector<float>::difference_type>(image_size) * static_cast<std::vector<float>::difference_type>(output_cursor)
                        );
                        input_cursor ++;
                        output_cursor ++;
                    }
                } else {
                    input_cursor +=will_decode;
                    output_cursor +=will_decode;
                }
                decode_time += clock.ms_elapsed();
            }
        } catch(queue_closed) {
        }
        myself->logger_.log_info(fmt(
            "[decoder          %02d/%02d]: {fetch_requet:%.5f}-->--{fetch_images:%.5f}-->--[process:%.5f]-->--{submit:%.5f}",
            decoder_id, num_decoders, fetch_reqs.value(), fetch_imgs.value(), process.value(), submit.value()
        ));
    }
public:
    thread_safe_queue<prefetch_msg*>& prefetch_queue() { return prefetch_queue_; }
    
    image_provider(const image_provider_opts& opts, inference_msg_pool* pool,
                   abstract_queue<inference_msg*>* request_queue, logger_impl& logger) 
        : data_provider(pool, request_queue), prefetch_queue_(opts.prefetch_queue_size_),
          opts_(opts), logger_(logger) {
        opts_.data_dir_ = fs_utils::normalize_path(opts_.data_dir_);
        if (!fs_utils::read_cache(opts_.data_dir_, file_names_)) {
            logger_.log_info("[image_provider        ]: found " + S(file_names_.size()) +  " image files in file system.");
            fs_utils::get_image_files(opts_.data_dir_, file_names_);
            if (!file_names_.empty()) {
                if (!fs_utils::write_cache(opts_.data_dir_, file_names_)) {
                     logger_.log_warning("[image_provider        ]: failed to write file cache.");
                }
            }
        } else {
            logger_.log_info("[image_provider        ]: read " + S(file_names_.size()) +  " from cache.");
            if (file_names_.empty()) { 
                logger_.log_warning("[image_provider        ]: found empty cache file. Please, delete it and restart DLBS. ");
            }
        }
        if (file_names_.empty()) {
            logger_.log_error("[image_provider        ]: no input data found, exiting.");
        }
        fs_utils::to_absolute_paths(opts_.data_dir_, file_names_);
        prefetchers_.resize(opts_.num_prefetchers_, nullptr);
        decoders_.resize(opts_.num_decoders_, nullptr);
    }
    
    virtual ~image_provider() {
        for (size_t i=0; i<prefetchers_.size(); ++i)
            if (prefetchers_[i]) delete prefetchers_[i];
        for (size_t i=0; i<decoders_.size(); ++i)
            if (decoders_[i]) delete decoders_[i];
    }
    
    void stop() override {
        data_provider::stop();
        prefetch_queue_.close();
    }
    
    void run() override {
        // Run prefetch workers
        for (size_t i=0; i<prefetchers_.size(); ++i) {
            prefetchers_[i] = new std::thread(&(image_provider::prefetcher_func), this, i, prefetchers_.size());
        }
        for (size_t i=0; i<decoders_.size(); ++i) {
            decoders_[i] = new std::thread(&(image_provider::decoder_func), this, i, decoders_.size());
        }
        // Wait
        for (auto& prefetcher : prefetchers_)
            if (prefetcher->joinable()) prefetcher->join();
        for (auto& decoder : decoders_)
            if (decoder->joinable()) decoder->join();
        // Clean prefetch queue
        std::vector<prefetch_msg*> queue_content;
        prefetch_queue_.empty_queue(queue_content);
        for (size_t i=0; i<queue_content.size(); ++i)
            delete queue_content[i];
    }
    
    };
namespace tests {
    namespace image_provider_tests {
        void benchmark_prefetch_readers();
        void benchmark_data_provider();
        float benchmark_crop(cv::Mat& img, const cv::Rect& crop_region=cv::Rect(10, 10, 256, 256), const int niters=100);
        float benchmark_resize(cv::Mat& img, const cv::Size& new_size=cv::Size(256, 256), const int niters=100);
        float benchmark_tovector(cv::Mat& img, const int niters=40000);
        float benchmark_tofloatmat(cv::Mat& img, const int niters=100);
        void read_image();
        
        
        void benchmark_prefetch_readers() {
            logger_impl logger;
            image_provider_opts opts;
            opts.data_dir_ = "/home/serebrya/data/";
            opts.num_prefetchers_ = 4;
            opts.num_decoders_ = 1;
            opts.prefetch_batch_size_=64;
            opts.prefetch_queue_size_=32;
            image_provider provider(opts, nullptr, nullptr, logger);
            provider.start();
            // N warmup iterations
            std::cout << "Running warmup iterations" << std::endl;
            for (int i=0; i<50; ++i) {
                prefetch_msg *imgs = provider.prefetch_queue().pop();
                delete imgs;
            }
            // N benchmark iterations
            std::cout << "Running benchmark iterations" << std::endl;
            timer t;
            size_t num_images(0);
            for (int i=0; i<100; ++i) {
                prefetch_msg *imgs = provider.prefetch_queue().pop();
                num_images += imgs->num_images();
                delete imgs;
            }
            const float throughput = 1000.0 * num_images / t.ms_elapsed();
            provider.stop();
            std::cout << "num_readers=" << opts.num_prefetchers_ << ", throughput=" << throughput << std::endl;
            
            provider.join();
        }
        void benchmark_data_provider() {
            logger_impl logger;
            image_provider_opts opts;
            opts.data_dir_ = "/home/serebrya/data/";
            opts.resize_method_ = "crop";
            opts.num_prefetchers_ = 16;
            opts.num_decoders_ = 8;
            opts.prefetch_batch_size_=64;
            opts.prefetch_queue_size_=64;
            opts.height_ = opts.width_ = 225;
            
            inference_msg_pool pool(10, opts.prefetch_batch_size_, opts.prefetch_batch_size_*3*opts.height_*opts.width_, 100);
            thread_safe_queue<inference_msg*> request_queue;
                   
            image_provider provider(opts, &pool, &request_queue, logger);
            provider.start();
            // N warmup iterations
            std::cout << "Running warmup iterations" << std::endl;
            for (int i=0; i<50; ++i) {
                inference_msg *imgs = request_queue.pop();
                pool.release(imgs);
            }
            // N benchmark iterations
            std::cout << "Running benchmark iterations" << std::endl;
            timer t;
            size_t num_images(0);
            for (int i=0; i<100; ++i) {
                inference_msg *imgs = request_queue.pop();
                pool.release(imgs);
                num_images += opts.prefetch_batch_size_;
            }
            const float throughput = 1000.0 * num_images / t.ms_elapsed();
            pool.close();
            provider.stop();
            std::cout << "num_readers=" << opts.num_prefetchers_ << ", throughput=" << throughput << std::endl;
            
            provider.join();
        }
        
        /** Crop     [3, 525, 700] -> [3, 256, 256] is ~ 3e-5 ms seems like just header change
         *  Resize   [3, 525, 700] -> [3, 256, 256] is ~ 1 ms  (cv::INTER_LINEAR)
         *  Tovector [3, 256, 256] -> [3*256*256]   is ~ 0.16 ms with crop first
         *  Tovector [3, 256, 256] -> [3*256*256]   is ~ 0.15 ms with resize first
         */
        float benchmark_crop(cv::Mat& img, const cv::Rect& crop_region, const int niters) {
            timer start;
            for (int i=0; i<niters; ++i)
                cv::Mat cropped = img(crop_region);
            return start.ms_elapsed() / niters;
        }
        float benchmark_resize(cv::Mat& img, const cv::Size& new_size, const int niters) {
            cv::Mat resized;
            timer start;
            for (int i=0; i<niters; ++i)
                cv::resize(img, resized, new_size, 0, 0, cv::INTER_LINEAR);
            return start.ms_elapsed() / niters;
        }
        __attribute__((noinline))
        float benchmark_tovector(cv::Mat& img, const int niters) {
            
            cv::Mat sized;
            sized = img(cv::Rect(10, 10, 256, 256));
            //cv::resize(img, sized, cv::Size(256,256), 0, 0, cv::INTER_LINEAR);
                
            std::vector<float> vec(static_cast<size_t>(sized.channels()*sized.rows*sized.cols));
            timer start;
            for (int i=0; i<niters; ++i) {
                vec.assign(sized.begin<float>(), sized.end<float>());
            }
            return start.ms_elapsed() / niters;
            
            //return 0.0;
        }
        float benchmark_tofloatmat(cv::Mat& img, const int niters) {
            /*
            cv::Mat cropped = img(cv::Rect(10, 10, 256, 256));
            cv::resize(img, cropped, new_size, 0, 0, cv::INTER_LINEAR);
            cv::Mat_<float> target;
            timer start;
            for (int i=0; i<niters; ++i)
                img.convertTo(target, CV_32FC3);
            return start.ms_elapsed() / niters;
            */
            return 0;
        }
        //
        void read_image() {
            // Read image
            std::cout << "Reading image ...\n";
            cv::Mat img = cv::imread("/home/serebrya/major.jpg");
            std::cout << "Original image dims: [" << img.channels() << ", " << img.rows << ", " << img.cols << "]\n";
            std::cout << "Is continious: " << img.isContinuous() << ", depth=" << img.depth() << ", type=" << img.type() << std::endl;
            std::cout << (img.dataend - img.datastart) << std::endl;
            // Crop image
            //const auto mean_crop_time = benchmark_crop(img);
            //std::cout << "Average crop time is " << mean_crop_time << " ms" << std::endl;
            // Resize image
            //const auto mean_resize_time = benchmark_resize(img);
            //std::cout << "Average resize time is " << mean_resize_time << " ms" << std::endl;
            // Convert to vector
            const auto mean_tovector_time = benchmark_tovector(img);
            std::cout << "Average tovector time is " << mean_tovector_time << " ms" << std::endl;
            // Convert to float matrix
            //const auto mean_tofloatmat_time = benchmark_tofloatmat(img);
            //std::cout << "Average tofloatmat time is " << mean_tofloatmat_time << " ms" << std::endl;
        }
    }
}
#endif

#endif
