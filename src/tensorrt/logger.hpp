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

#ifndef DLBS_TENSORRT_BACKEND_LOGGER
#define DLBS_TENSORRT_BACKEND_LOGGER

#include <mutex>
#include <iostream>
#include <vector>
#include <algorithm>

#include <NvInfer.h>
using namespace nvinfer1;


//A simple logger for TensorRT library.
class logger_impl : public ILogger {
private:
    std::mutex m_;           //!< Mutex that guards output stream.
    std::ostream& ostream_; //!< Output logging stream.
public:
    explicit logger_impl(std::ostream& ostream=std::cout) : ostream_(ostream) {}
    /**
     * @brief Log intermidiate performance results This is usefull to estimate jitter online or when
     * running long lasting benchmarks.
     * 
     * @param times[in] A vector of individual batch times in milliseconds collected so far. We are interested
     * in times starting from \p iter_index value.
     * @param iter_index An index of the last iteration start
     * @param data_size[in] Data size as number of instances for which individial \p times are reported. In
     * most cases this is the same as effective batch size.
     * @param out[in] A stream to write statistics.
     */
    void log_progress(const std::vector<float>& times, const int iter_index,
                      const int data_size, const std::string& key_prefix) {
        if (times.empty()) return;
        const float sum = std::accumulate(times.begin(), times.end(), 0.0f);
        const float mean = sum / times.size();
        const float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
        const float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
        const float throughput = 1000.0 * data_size / mean;
        std::lock_guard<std::mutex> lock(m_);
        ostream_ << "__results." << key_prefix << "progress__=[" << mean << ", " << stdev << ", " << throughput << "]" << std::endl;
    }

    /** 
     * @brief Log final benchmark results to a standard output.
     * 
     * @param times[in,out] A vector of individual batch times in milliseconds. This vector will be sorted
     * by this method. Each element is a time in seconds it took to process \p data_size input instances.
     * @param data_size[in] Data size as number of instances for which individial \p times are reported. In
     * most cases this is the same as effective batch size.
     * @param key_prefix[in] A key prefix for a key. Identifies what is to be logged. It can be empty
     * to log inference times not taking into account CPU <--> GPU data transfers or 'total_' to log
     * total inference times including data transfers to and from GPU.
     * @param report_times[in] If true, write the content of \p times as well.
     * @param out[in] A stream to write statistics.
     * 
     * This method logs the following keys:
     *   results.${key_prefix}time           A mean value of \p times vector.
     *   results.${key_prefix}throughput     Throughput - number of input instances per second.
     *   results.${key_prefix}time_data      Content of \p times.
     *   results.${key_prefix}time_stdev     Standard deviation of \p times vector.
     *   results.${key_prefix}time_min       Minimal time in \p times vector.
     *   results.${key_prefix}time_max       Maximal time in \p times vector.
     */
    void log_final_results(std::vector<float>& times, const size_t data_size,
                           const std::string& key_prefix="", const bool report_times=true) {
        if (times.empty()) return;
        std::sort(times.begin(), times.end());
        const float sum = std::accumulate(times.begin(), times.end(), 0.0f);
        const float mean = sum / times.size();
        const float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
        const float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
        const float throughput = 1000.0 * data_size / mean;

        std::lock_guard<std::mutex> lock(m_);
        ostream_ << "__results." << key_prefix << "time__= " << mean  << "\n";
        ostream_ << "__results." << key_prefix << "throughput__= " << throughput  << "\n";
        if (report_times) {
            ostream_ << "__results." << key_prefix << "time_data__=[";
            for (std::vector<float>::size_type i=0; i<times.size(); ++i) {
                if (i != 0) { ostream_ << ","; }
                ostream_ << times[i];
            }
            ostream_ << "]" << "\n";
        }
        ostream_ << "__results." << key_prefix << "time_stdev__= " << stdev  << "\n";
        ostream_ << "__results." << key_prefix << "time_min__= " << times.front()  << "\n";
        ostream_ << "__results." << key_prefix << "time_max__= " << times.back()  << "\n";
        ostream_ << std::flush;
    }
  
    // Print engine bindings (input/output blobs)
    void log_bindings(ICudaEngine* engine) {
        std::lock_guard<std::mutex> lock(m_);
        const auto num_bindings = engine->getNbBindings();
        ostream_ << "engine::number of bindings = " << num_bindings << "\n";
        for (auto i=0; i<num_bindings; ++i) {
            ostream_ << "engine::binding index = " << i << ", name = " << engine->getBindingName(i) << ", is input = " << engine->bindingIsInput(i);
#if NV_TENSORRT_MAJOR >= 3
            const Dims shape = engine->getBindingDimensions(i);
            ostream_ << ", shape=[";
            for (int j=0; j<shape.nbDims; ++j) {
                if (j != 0) {
                    ostream_ << ", ";
                }
                ostream_ << shape.d[j];
            }
            ostream_ << "]" << "\n";
#else
            const Dims3 dims = engine->getBindingDimensions(i);
            ostream_ << ", shape=[" << dims.c << ", " << dims.h << ", " << dims.w << "]" << "\n";
#endif
        }
        ostream_ << std::flush;
    }

    void log(Severity severity, const char* msg) override { log_internal(severity, msg); }

    template <typename T> void log_internal_error(const T& msg) { log_internal(ILogger::Severity::kINTERNAL_ERROR, msg); }
    template <typename T> void log_error(const T& msg) { log_internal(ILogger::Severity::kERROR, msg); }
    template <typename T> void log_warning(const T& msg) { log_internal(ILogger::Severity::kWARNING, msg); }
    template <typename T> void log_info(const T& msg) { log_internal(ILogger::Severity::kINFO, msg); }
private:
    template <typename T>
    void log_internal(Severity severity, const T& msg) {
        std::lock_guard<std::mutex> lock(m_);
        ostream_ << time_stamp() << " "  << log_levels_[severity] << " "  << msg << std::endl;
        if (severity == ILogger::Severity::kINTERNAL_ERROR || severity == ILogger::Severity::kERROR) {
            exit(1);
        }
    }
    
    std::string time_stamp() {
        time_t rawtime;
        time (&rawtime);
        struct tm * timeinfo = localtime(&rawtime);
        // YYYY-mm-dd HH:MM:SS    19 characters
        char buffer[20];
        const auto len = strftime(buffer,sizeof(buffer),"%F %T",timeinfo);
        return (len > 0 ? std::string(buffer) : std::string(19, ' '));
    }
private:
    std::map<Severity, std::string> log_levels_ = {
        {ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR"},
        {ILogger::Severity::kERROR,          "         ERROR"},
        {ILogger::Severity::kWARNING,        "       WARNING"},
        {ILogger::Severity::kINFO,           "          INFO"}
    };
  
};

#endif