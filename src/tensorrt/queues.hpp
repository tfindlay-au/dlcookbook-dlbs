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
#ifndef DLBS_TENSORRT_BACKEND_QUEUES
#define DLBS_TENSORRT_BACKEND_QUEUES

#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>

template <typename T>
class abstract_queue {
protected:
    std::mutex m_;
public:
    virtual T pop() = 0;
    virtual void push(const T& item) = 0;
};

/**
 * @brief A simple infinite queue with one non-removeable element. 
 * 
 * There's only one element in a queue that can be changed with  infinite_queue:push
 * and infinite_queue:pop method returns it without removing this element from the 
 * queue.
 */
template <typename T>
class infinite_queue : public abstract_queue<T> {
private:
    using abstract_queue<T>::m_;
    T item_;
public:
    infinite_queue(const T& item) {
        item_ = item;
    }
    void push(const T& item) override {
        std::lock_guard<std::mutex> lock(m_);
        item_ = item;
    }
    
    T pop() override{
        std::lock_guard<std::mutex> lock(m_);
        return item_;
    }
};

/**
 * @brief A thread safe queue that can be used to exchange data between
 * various threads.
 */
template <typename T>
class thread_safe_queue : public abstract_queue<T> {
private:
    using abstract_queue<T>::m_;
    std::condition_variable cond_;
    std::queue<T> queue_;
public:
    /**
     * @brief Pushes the data into the queue. 
     * @param item Item to push into the queue.
     * 
     * Depending of template type, this methid may or may not create copy.
     * The TensorRT benchmarking backed uses pointers, so no copies are created.
     */
    void push(const T& item) override {
        std::lock_guard<std::mutex> lock(m_);
        queue_.push(item);
        cond_.notify_one();
    }
    /**
     * @brief Returns front element from the queue. This is a blocking call.
     */
    T pop() override{
        std::unique_lock<std::mutex> lock(m_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        return item;
    }
};


namespace tests {
    namespace infinite_queue_tests {
        void worker(infinite_queue<int*>& q, long& counter) {
            while (q.pop() != nullptr) {
                counter ++;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        void test() {
            infinite_queue<int*> q(nullptr);
            int v=1;
            q.push(&v);
            long counter(0);
            std::thread w(worker, std::ref(q), std::ref(counter));
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            q.push(nullptr);
            w.join();
            std::cout << "counter=" << counter << std::endl;
        }
    }
}

#endif