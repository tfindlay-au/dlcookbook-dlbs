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

#include "core/logger.hpp"
#include "core/utils.hpp"

#include <boost/program_options.hpp>
#include <thread>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace po = boost::program_options;

/**
 * @brief Convert a subset of images into a tensor representation. This function runs in its own thread.
 * @param input_files List of all input files to convert. It's relative to \p input_dir.
 * @param input_dir Input directory containing input files.
 * @param output_dir Output directory. Directory structure will be the same as in \p input_dir.
 * @param num_shards Total number of parallel workers converting a dataset (word size).
 * @param my_shard My index in the list of workers (my rank).
 * @param img_size Spatial dimensions of images (3 * img_size * img_size), i.e. a three channel image
 *                 of square shape.
 * @param logger A thread safe logger.
 */
void convert(std::vector<std::string>& input_files, const std::string input_dir, const std::string output_dir,
             const size_t num_shards, const size_t my_shard, const size_t img_size, logger_impl& logger) {
#ifdef HAS_OPENCV
    std::vector<float> tensor(3 * img_size * img_size);
    sharded_vector<std::string> my_files(input_files, num_shards, my_shard, true);
    const int nchannels = 3;
    
    logger.log_info(fmt("Thread %d: shard_begin=%d, shard_length=%d", my_shard, my_files.shard_begin(), my_files.shard_length()));
    long nprocessed(0);
    timer tm;
    while(my_files.has_next()) {
        const std::string file_name = my_files.next();
        cv::Mat img = cv::imread(input_dir + file_name);
        if (!img.data) {
            logger.log_warning(fmt("Thread %d: bad input file %s", int(my_shard), file_name.c_str()));
            continue;
        }
        cv::resize(img, img, cv::Size(img_size, img_size), 0, 0, cv::INTER_LINEAR);
            
        tensor.assign(img.begin<float>(), img.end<float>());
        const std::string output_file_name = output_dir + file_name;
        fs_utils::mk_dir(fs_utils::parent_dir(output_file_name));

        std::ofstream out(output_file_name.c_str());
        if (!out) {
            logger.log_warning(fmt("Thread %d: cannot write file %s", int(my_shard), output_file_name.c_str()));
            continue;
        }

        out.write((const char*)&nchannels, sizeof(int));
        out.write((const char*)&img_size, sizeof(int));
        out.write((const char*)&img_size, sizeof(int));
        out.write((const char*)tensor.data(), tensor.size()*sizeof(float));
        
        nprocessed ++;
    }
    const float throughput = 1000.0 * nprocessed / tm.ms_elapsed();
    logger.log_info(fmt("Thread %d: throughput %f images/sec", my_shard, throughput));
#endif
}


/**
 * @brief Main entry point.
 * Usage:
 * images2tensors --input_dir=DIR --output_dir=DIR --size=227 --shuffle  --nimages=0 --nthreads=1 
 */
int main(int argc, char **argv) {
    //
    logger_impl logger(std::cout);
    std::string input_dir,
                output_dir;
    int size,
        nthreads,
        nimages;
    bool shuffle;
#ifndef HAS_OPENCV
    std::cerr << "The images2tensor tool was compiled without OpenCV support and hence cannot load and resize images." << std::endl
              << "It does not support generating artificial datasets for benchmarking purposes. Open a new issue on" << std::endl
              << "GitHub and we will add this functionaity" << std::endl;
    logger.log_error("Can not do anything without OpenCV support.");
#endif
    
    // Parse command line options
    po::options_description opt_desc("Images2Tensors");
    po::variables_map var_map;
    opt_desc.add_options()
        ("help", "Print help message")
        ("input_dir", po::value<std::string>(&input_dir)->required(), 
            "Input directory. This directory must exist and must contain images (jpg, jpeg) in that directory "
            "or one of its sub-directories. ImageNet directory with raw images is one example of a valid directory."
        )
        ("output_dir", po::value<std::string>(&output_dir)->required(),
            "Output directory. Directory that will have exactly the same structure as input directory. Each input "
            "file will get same relative path, will have same name and extension. Even though file extension will "
            "remain the same, the content will be different. It will not be a valid image files."
        )
        ("size", po::value<int>(&size)->default_value(227), 
            "Resize images to this size. Output images will have square shape [3, size, size]."
        )
        ("shuffle",  po::bool_switch(&shuffle)->default_value(false),
            "Shuffle list if images. Usefull with combination --nimages to convert only a small random subset."
        )
        ("nimages", po::value<int>(&nimages)->default_value(0),
            "If nimages > 0, only convert this number of images. Use --shuffle to randomly shuffle list of images "
            "with this option."
        )
        ("nthreads", po::value<int>(&nthreads)->default_value(1),
            "Use this number of threads to convert images. This will significantly increase overall throughput. "
            "On my dev box, single-threaded performance is ~300-400 images/sec."
        );
    try {
        po::store(po::parse_command_line(argc, argv, opt_desc), var_map);
        if (var_map.count("help")) {
            std::cout << opt_desc << std::endl;
            return 0;
        }
        po::notify(var_map);
    } catch(po::error& e) {
        logger.log_warning(e.what());
        std::cout << opt_desc << std::endl;
        logger.log_error("Cannot recover from previous errors");
    }

    // Get list of input files
    input_dir = fs_utils::normalize_path(input_dir);
    std::vector<std::string> file_names;
    fs_utils::get_image_files(input_dir, file_names);
    if (file_names.empty())
        logger.log_error(fmt("images2tensors: No input files found in '%s'", input_dir.c_str()));
    logger.log_info(fmt("images2tensors: have found %d images in '%s'", file_names.size(), input_dir.c_str()));
    //
    if (shuffle) {
        logger.log_info("images2tensors: shuffling list of file names");
        std::random_shuffle(file_names.begin(), file_names.end());
    }
    if (nimages > 0 && nimages < file_names.size()) {
        logger.log_info(fmt("images2tensors: Reducing number of images to convert to %d", nimages));
        file_names.resize(nimages);
    }
    
    // Convert and write
    output_dir = fs_utils::normalize_path(output_dir);
    std::vector<std::thread*> workers(nthreads, nullptr);
    timer tm;
    for (int i=0; i<nthreads; ++i) {
        workers[i] = new std::thread(convert, std::ref(file_names), input_dir, output_dir, nthreads, i, size, std::ref(logger));
    }
    for (int i=0; i<nthreads; ++i) {
        workers[i]->join();
        delete workers[i];
    }
    const float throughput = 1000.0 * file_names.size() / tm.ms_elapsed();
    logger.log_info(fmt("images2tensors: total throughput %f images/sec", throughput));
    return 0;
}
