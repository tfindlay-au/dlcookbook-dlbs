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

#include <iostream>
#include <vector>
#include <algorithm>

#include <NvInfer.h>
using namespace nvinfer1;


//A simple logger for TensorRT library.
class tensorrt_logger : public ILogger {
public:
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
    static void log_progress(const std::vector<float>& times,
                             const int iter_index,
                             const int data_size,
                             const std::string& key_prefix,
                             std::ostream& out=std::cout) {
      if (times.empty()) return;
      const float sum = std::accumulate(times.begin(), times.end(), 0.0f);
      const float mean = sum / times.size();
      const float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
      const float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
      const float throughput = 1000.0 * data_size / mean;
      out << "__results." << key_prefix << "progress__=[" << mean << ", " << stdev << ", " << throughput << "]" << std::endl;
      out << std::flush;
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
  static void log_final_results(std::vector<float>& times,
                                const int data_size,
                                const std::string& key_prefix="",
                                const bool report_times=true,
                                std::ostream& out=std::cout) {
    if (times.empty()) return;
    std::sort(times.begin(), times.end());
    const float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    const float mean = sum / times.size();
    const float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
    const float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
    const float throughput = 1000.0 * data_size / mean;
        
    out << "__results." << key_prefix << "time__= " << mean  << std::endl;
    out << "__results." << key_prefix << "throughput__= " << throughput  << std::endl;
    if (report_times) {
      out << "__results." << key_prefix << "time_data__=[";
      for (int i=0; i<times.size(); ++i) {
        if (i != 0) { out << ","; }
        out << times[i];
      }
      out << "]" << std::endl;
    }
        
    out << "__results." << key_prefix << "time_stdev__= " << stdev  << std::endl;
    out << "__results." << key_prefix << "time_min__= " << times.front()  << std::endl;
    out << "__results." << key_prefix << "time_max__= " << times.back()  << std::endl;
    out << std::flush;
  }
  
  // Print engine bindings (input/output blobs)
  void log_bindings(ICudaEngine* engine) {
    const auto num_bindings = engine->getNbBindings();
    std::cout << "engine::number of bindings = " << num_bindings << std::endl;
    for (auto i=0; i<num_bindings; ++i) {
      std::cout << "engine::binding index = " << i << ", name = " << engine->getBindingName(i) << ", is input = " << engine->bindingIsInput(i);
#if NV_TENSORRT_MAJOR >= 3
      const Dims shape = engine->getBindingDimensions(i);
      std::cout << ", shape=[";
      for (int i=0; i<shape.nbDims; ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << shape.d[i];
      }
      std::cout << "]" << std::endl;
#else
      const Dims3 dims = engine->getBindingDimensions(i);
      std::cout << ", shape=[" << dims.c << ", " << dims.h << ", " << dims.w << "]" << std::endl;
#endif
    }
  }

  /**
   * severity: [kINTERNAL_ERROR, kERROR, kWARNING, kINFO]
   */
  virtual void log(Severity severity, const char* msg) override {
    std::cerr << time_stamp() << " " 
              << log_levels_[severity] << " " 
              << msg << std::endl;
    if (severity == ILogger::Severity::kINTERNAL_ERROR || severity == ILogger::Severity::kERROR) {
      exit(1);
    }
  }
  void log_internal_error(const char* msg) { log(ILogger::Severity::kINTERNAL_ERROR, msg); }
  void log_error(const char* msg) { log(ILogger::Severity::kERROR, msg); }
  void log_warning(const char* msg) { log(ILogger::Severity::kWARNING, msg); }
  void log_info(const char* msg) { log(ILogger::Severity::kINFO, msg); }
  
  void log_internal_error(const std::string& msg) { log_internal_error(msg.c_str()); }
  void log_error(const std::string& msg) { log_error(msg.c_str()); }
  void log_warning(const std::string& msg) { log_warning(msg.c_str()); }
  void log_info(const std::string& msg) { log_info(msg.c_str()); }
private:
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