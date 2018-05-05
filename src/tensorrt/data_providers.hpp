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

#include "infer_task.hpp"

/**
 * @brief Base abstract class for all data providers.
 */
template <typename T>
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
    task_pool<T>* task_pool_;         //!< [input]  A pool of free tasks that can be reused to submit infer requests.
    abstract_queue<T*>* data_queue_;  //!< [output] An output data queue containing requests with real data.
    std::atomic_bool stop_;           //!< The 'stop' flag indicating internal thread must stop.
public:
    data_provider(task_pool<T>* pool, abstract_queue<T*>* data_queue) 
        : task_pool_(pool), data_queue_(data_queue), stop_(false) {
    }
    virtual ~data_provider() {
        if (internal_thread_) delete internal_thread_;
    }
    //!< Starts internal thread to load data into the data queue.
    void start() {
        internal_thread_ = new std::thread(&data_provider::thread_func, this);
    }
    //!< Requests to stop internal thread and returns without waiting.
    void stop() {
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
template <typename T>
class synthetic_data_provider : public data_provider<T> {
private:
    using data_provider<T>::task_pool_;
    using data_provider<T>::data_queue_;
    using data_provider<T>::stop_;
public:
    synthetic_data_provider(task_pool<T>* pool, abstract_queue<T*>* data_queue) 
        : data_provider<T>(pool, data_queue) {
    }
    void run() override {
        while(true) {
            T* task = task_pool_->get();
            if (stop_ || task == nullptr)
                break;
            data_queue_->push(task);
        }
    }
};

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>

class raw_img_data_provider : public data_provider<infer_task> {
private:
    using T=infer_task;
    using data_provider<T>::task_pool_;
    using data_provider<T>::data_queue_;
    using data_provider<T>::stop_;
public:
    raw_img_data_provider(task_pool<T>* pool, abstract_queue<T*>* data_queue) 
        : data_provider<T>(pool, data_queue) {
    }
};
namespace tests {
    namespace raw_img_data_provider_tests {
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
