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

#include <exception>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>

class queue_closed : public std::exception {
    const std::string msg = "Queue was closed while performing requested operation.";
public:
    queue_closed() {}
    const char* what() const noexcept override { return msg.c_str(); }
};

template <typename T>
class abstract_queue {
protected:
    std::mutex m_;
    std::condition_variable push_evnt_;
    std::condition_variable pop_evnt_;
    std::atomic_bool closed_;
public:
    void close() { 
        std::unique_lock<std::mutex> lock(m_);
        closed_ = true;
        push_evnt_.notify_all();
        pop_evnt_.notify_all();
    }
    bool is_closed() const { return closed_; }
    
    virtual T pop() throw (queue_closed) = 0;
    virtual void push(const T& item) throw (queue_closed) = 0;
    virtual void empty_queue(std::vector<T>& queue_content) = 0;
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
    using abstract_queue<T>::closed_;
    T item_;
    bool emptied_ = false;
public:
    explicit infinite_queue(const T& item) {
        closed_ = false;
        item_ = item;
    }
    void push(const T& item) throw (queue_closed) override {
        if (closed_) 
            throw queue_closed();
        std::lock_guard<std::mutex> lock(m_);
        item_ = item;
    }
    
    T pop() throw (queue_closed) override {
        if (closed_)
            throw queue_closed();
        std::lock_guard<std::mutex> lock(m_);
        return item_;
    }
    
    void empty_queue(std::vector<T>& queue_content) override {
        if (!closed_ || emptied_)
            return;
        std::lock_guard<std::mutex> lock(m_);
        queue_content.clear();
        queue_content.push_back(item_);
        emptied_ = true;
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
    using abstract_queue<T>::closed_;
    using abstract_queue<T>::push_evnt_;
    using abstract_queue<T>::pop_evnt_;
    std::queue<T> queue_;
    size_t max_size_;
public:
    explicit thread_safe_queue(const size_t max_size=0) : max_size_(max_size) {
        closed_ = false;
    }
    thread_safe_queue(const thread_safe_queue& q) = delete;

    /**
     * @brief Pushes the data into the queue. 
     * @param item Item to push into the queue.
     * 
     * Depending of template type, this methid may or may not create copy.
     * The TensorRT benchmarking backed uses pointers, so no copies are created.
     */
    void push(const T& item) throw (queue_closed) override {
        if (closed_)
            throw queue_closed();
        std::unique_lock<std::mutex> lock(m_);
        pop_evnt_.wait(
            lock, 
            [this] { return (closed_ || max_size_ <=0 || queue_.size() < max_size_); }
        );
        if (closed_)
            throw queue_closed();
        queue_.push(item);
        push_evnt_.notify_one();
    }
    /**
     * @brief Returns front element from the queue. This is a blocking call.
     */
    T pop() throw (queue_closed) override {
        if (closed_)
            throw queue_closed();
        std::unique_lock<std::mutex> lock(m_);
        push_evnt_.wait(
            lock, 
            [this] { return closed_ || !queue_.empty(); }
        );
        if (closed_)
            throw queue_closed();
        T item = queue_.front();
        queue_.pop();
        pop_evnt_.notify_one();
        return item;
    }
    
    void empty_queue(std::vector<T>& queue_content) override {
        if (!closed_)
            return;
        std::unique_lock<std::mutex> lock(m_);
        queue_content.clear();
        while (!queue_.empty()) {
            queue_content.push_back(queue_.front());
            queue_.pop();
        }
        if (!queue_content.empty())
            std::reverse(queue_content.begin(), queue_content.end());
    }
};


namespace tests {
    namespace queue_tests {
        void consumer(abstract_queue<int>* queue, long& counter);
        void provider(abstract_queue<int>* queue, long& counter);
        void test_infinite_queue();
        
        
        void consumer(abstract_queue<int>* queue, long& counter) {
            try {
                while (true) {
                    queue->pop();
                    counter ++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            } catch(queue_closed) {
            }
        }
        void provider(abstract_queue<int>* queue, long& counter) {
            try {
                while (true) {
                    queue->push(1);
                    counter ++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            } catch(queue_closed) {
            }
        }

        void test_infinite_queue() {
            infinite_queue<int> q(1);
            long consumer_counter(0), provider_counter(0);
            std::thread c(consumer, &q, std::ref(consumer_counter));
            std::thread p(provider, &q, std::ref(provider_counter));
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            q.close();
            c.join();
            p.join();
            std::cout << "consumer_counter=" << consumer_counter << std::endl;
            std::cout << "provider_counter=" << provider_counter << std::endl;
        }
    }
}

#endif