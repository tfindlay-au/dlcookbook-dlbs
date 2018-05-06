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
    std::string data_dir_;

    std::string resize_method_ = "crop";
    int height_ = 0;
    int width_ = 0;
    
    int num_prefetchers_ = 0;
    int num_decoders_ = 0;
    
    int prefetch_batch_size_ = 0;
    int prefetch_queue_size_ = 0;
    
    resize_method get_resize_method() const {
        return resize_method_ == "resize"? resize_method::resize : resize_method::crop;
    }
    void log() {
        std::cout << "[image_provider_opts] "
                  << "data_dir=" << data_dir_ << ", resize_method=" << resize_method_ << ", height=" << height_
                  << ", width=" << width_ << ", num_prefetchers=" << num_prefetchers_
                  << ", num_decoders=" << num_decoders_ << ", prefetch_batch_size=" << prefetch_batch_size_
                  << ", prefetch_queue_size=" << prefetch_queue_size_
                  << std::endl;
                  
    }
};

/**
 * @brief Sharded vector iterates forever over provided chunk. For instance, we can have
 * a vector of file names of images. We then can have multiple image readers that will read
 * images from their own chunk.
 */
template <typename T>
class sharded_vector {
private:
    std::vector<T>* vec_;
    int first_idx_;
    int length_;
    int pos_;
public:
    sharded_vector(std::vector<T>& vec, const int first_idx, const int length)
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
    const int num_images() const { return images_.size(); }
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
private:
    static void prefetcher_func(image_provider* myself, const int id) {
        // Find out images I am responsible for.
        int shard_pos(0), shard_length(0);
        get_my_shard(myself->file_names_.size(), myself->prefetchers_.size(), id, shard_pos, shard_length);
        sharded_vector<std::string> my_files(myself->file_names_, shard_pos, shard_length);
        prefetch_msg *msg = new prefetch_msg();
        try {
            while (!myself->stop_) {
                const auto fname = my_files.next();
                cv::Mat img = cv::imread(fname);
                if (img.data == nullptr) {
                    std::cerr << "Error loading image from file: " << fname << std::endl;
                    continue;
                }
                msg->images_.push_back(img);
                if (msg->num_images() >= myself->opts_.prefetch_batch_size_) {
                    myself->prefetch_queue_.push(msg);
                    msg = new prefetch_msg();
                }
            }
        } catch(queue_closed) {
        }
        std::cout << "prefetcher (reader) " << id << " has shut down" << std::endl;
        delete msg;
    }
    
    static void decoder_func(image_provider* myself, const int id) {
        const int height(myself->opts_.height_),
                  width(myself->opts_.width_);
        const resize_method resizem = myself->opts_.get_resize_method();
        const int image_size = 3 * height * width;
        try {
            while(!myself->stop_) {
                // Get free task from the task pool
                inference_msg *infer_request = myself->inference_msg_pool_->get();
                // Get prefetched images
                prefetch_msg* raw_images = myself->prefetch_queue_.pop();
                // Crop and convert to float array. The input_ field in task
                // has the following shape: [BatchSize, NumChannels, Height, Width]
                // Every image will be [NumChannels, Height, Width] and num_images
                // below must be exactly BatchSize.
                const int num_images = raw_images->num_images();
                for (int i=0; i<num_images; ++i) {
                    cv::Mat img = raw_images->images_[i];
                    if (img.rows != height || img.cols != width) {
                        if (resizem == resize_method::resize || img.rows < height || img.cols < width) {
                            cv::resize(raw_images->images_[i], img, cv::Size(height, width), 0, 0, cv::INTER_LINEAR);
                        } else {
                            img = img(cv::Rect(0, 0, height, width));
                        }
                    }
                    std::copy(
                        img.begin<float>(),
                        img.end<float>(),
                        infer_request->input_.begin() + i*image_size
                    );
                }
                delete raw_images;
                // Push preprocessed images into the queue
                myself->request_queue_->push(infer_request);
            }
        } catch(queue_closed) {
        }
        std::cout << "decoder " << id << " has shut down" << std::endl;
    }
public:
    thread_safe_queue<prefetch_msg*>& prefetch_queue() { return prefetch_queue_; }
    
    image_provider(const image_provider_opts& opts, inference_msg_pool* pool,
                   abstract_queue<inference_msg*>* request_queue) 
        : data_provider(pool, request_queue), prefetch_queue_(opts.prefetch_queue_size_), opts_(opts) {
        read_file(opts_.data_dir_ + "/images.txt", file_names_);
        const int num_files_read = file_names_.size();
        std::cout << "Number of images: " << num_files_read << std::endl;
        for (int i=0; i<num_files_read; ++i) {
            file_names_[i] = opts_.data_dir_ + file_names_[i];
        }
        prefetchers_.resize(opts_.num_prefetchers_, nullptr);
        decoders_.resize(opts_.num_decoders_, nullptr);
    }
    
    virtual ~image_provider() {
        for (int i=0; i<prefetchers_.size(); ++i)
            if (prefetchers_[i]) delete prefetchers_[i];
        for (int i=0; i<decoders_.size(); ++i)
            if (decoders_[i]) delete decoders_[i];
    }
    
    void stop() override {
        data_provider::stop();
        prefetch_queue_.close();
    }
    
    void run() override {
        // Run prefetch workers
        const int num_prefetchers = prefetchers_.size();
        for (int i=0; i<num_prefetchers; ++i) {
            prefetchers_[i] = new std::thread(&(image_provider::prefetcher_func), this, i);
        }
        const int num_decoders = decoders_.size();
        for (int i=0; i<num_decoders; ++i) {
            decoders_[i] = new std::thread(&(image_provider::decoder_func), this, i);
        }
        // Wait
        for (auto& prefetcher : prefetchers_)
            if (prefetcher->joinable()) prefetcher->join();
        for (auto& decoder : decoders_)
            if (decoder->joinable()) decoder->join();
        // Clean prefetch queue
        std::vector<prefetch_msg*> queue_content;
        prefetch_queue_.empty_queue(queue_content);
        for (int i=0; i<queue_content.size(); ++i)
            delete queue_content[i];
    }
    
    };
namespace tests {
    namespace image_provider_tests {
        void benchmark_prefetch_readers() {
            image_provider_opts opts;
            opts.data_dir_ = "/home/serebrya/data/";
            opts.num_prefetchers_ = 4;
            opts.num_decoders_ = 1;
            opts.prefetch_batch_size_=64;
            opts.prefetch_queue_size_=32;
            image_provider provider(opts, nullptr, nullptr);
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
            int num_images(0);
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
            image_provider_opts opts;
            opts.data_dir_ = "/home/serebrya/data/";
            opts.resize_method_ = "crop";
            opts.num_prefetchers_ = 16;
            opts.num_decoders_ = 8;
            opts.prefetch_batch_size_=64;
            opts.prefetch_queue_size_=64;
            opts.height_ = opts.width_ = 225;
            
            inference_msg_pool pool(10, opts.prefetch_batch_size_*3*opts.height_*opts.width_, 100);
            thread_safe_queue<inference_msg*> request_queue;
                   
            image_provider provider(opts, &pool, &request_queue);
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
            int num_images(0);
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
        float benchmark_crop(cv::Mat& img, const cv::Rect& crop_region=cv::Rect(10, 10, 256, 256), const int niters=1000000) {
            timer start;
            for (int i=0; i<niters; ++i)
                cv::Mat cropped = img(crop_region);
            return start.ms_elapsed() / niters;
        }
        float benchmark_resize(cv::Mat& img, const cv::Size& new_size=cv::Size(256, 256), const int niters=100000) {
            cv::Mat resized;
            timer start;
            for (int i=0; i<niters; ++i)
                cv::resize(img, resized, new_size, 0, 0, cv::INTER_LINEAR);
            return start.ms_elapsed() / niters;
        }
        float benchmark_tovector(cv::Mat& img, const int niters=400000) {
            cv::Mat sized;
            if (true) {
                sized = img(cv::Rect(10, 10, 256, 256));
            } else {
                cv::resize(img, sized, cv::Size(256,256), 0, 0, cv::INTER_LINEAR);
            }
                
            std::vector<float> vec(sized.channels()*sized.rows*sized.cols);
            timer start;
            for (int i=0; i<niters; ++i)
                vec.assign(sized.begin<float>(), sized.end<float>());
            return start.ms_elapsed() / niters;
        }
        float benchmark_tofloatmat(cv::Mat& img, const int niters=100000) {
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
        void raw_img_data_provider_test() {
        }
    }
}
#endif

#endif
