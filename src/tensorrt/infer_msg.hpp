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
#ifndef DLBS_TENSORRT_BACKEND_INFER_MSG
#define DLBS_TENSORRT_BACKEND_INFER_MSG

#include "utils.hpp"
#include "queues.hpp"

/**
 * @brief A structure that contains input/output data for an inference task. It's better
 * to create a pool of these objects and reuse them.
 */
struct inference_msg {
    std::vector<float> input_;   //!< Input data of shape [BatchSize, ...]
    std::vector<float> output_;  //!< Input data of shape [BatchSize, ...]

    float batch_time_ = 0;       //!< Total batch time including CPU <-> GPU transfer overhead
    float infer_time_ = 0;       //!< Inference time excluding data transfer overhead
    
    int gpu_ = 0;                //!< GPU that processed this task.

    /**
     * @brief Construct and initialize inference task.
     * @param input_size Number of elements in an input tensor 
     * including batch dimension
     * @param output_size Number of elements in an output tensor
     * including batch dimension
     * @param randomize_input If true, randomly initialize input tensor.
     */
    inference_msg(const int input_size, const int output_size, const bool randomize_input=false) {
        input_.resize(input_size);
        output_.resize(output_size);
        if (randomize_input)
            random_input();
    }
    /**
     * @brief Fill input tensor with random data 
     * uniformly distributed in [0, 1]
     */
    void random_input() { fill_random(input_); }
};


/**
 * @brief Pool of task objects initialized to have correct storage size. This is used to
 * not allocate/deallocate memory during benchmarks.
 * To submit new infer request, fetch free task from this pool, initialize with your input
 * data and submit to a data queue. Once results is obtained, release the task by making it
 * avaialble for subsequent requests.
 */
class inference_msg_pool {
private:
    std::vector<inference_msg*> messages_;             //!< All allocated messages managed by the pool.
    thread_safe_queue<inference_msg*> free_messages_;  //!< Messages that are currently available.
    std::atomic_bool stop_;
public:
    inference_msg_pool(const int count, const int input_size, const int output_size,
                       const bool randomize_input=false) : stop_(false) {
        for (int i=0; i<count; ++i) {
            inference_msg *msg= new inference_msg(input_size, output_size, randomize_input);
            messages_.push_back(msg);
            free_messages_.push(msg);
        }
    }
    ~inference_msg_pool() {
        for (int i=0; i<messages_.size(); ++i) {
            delete messages_[i];
        }
    }
    //!< Get new task. The taks may or may not contain data from previous inference.
    //!< You should try to reuse memory allocated for this task.
    inference_msg* get() {
        return free_messages_.pop();
    }
    //!< Make this task object available for subsequent inferences.
    void release(inference_msg *msg) {
        free_messages_.push(msg);
    }
    //!< 'Stop' the pool. After this call, task_pool:get will be returning nullptr.
    void close() {
        free_messages_.close();
    }
};

#endif
