cmake_minimum_required(VERSION 2.6)
project(tensorrt_benchmarks)

add_compile_options(-std=c++11)

#add_compile_options(-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization)
#add_compile_options(-Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept)
#add_compile_options(-Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo)
#add_compile_options(-Wstrict-null-sentinel -Wswitch-default -Wundef -Werror -Wno-unused)
#add_compile_options(-Wstrict-overflow=5)

add_executable(tensorrt main.cpp)

FIND_PACKAGE(CUDA COMPONENTS cudart REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${CUDA_LIBRARIES})
endif()

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${Boost_LIBRARIES})
endif()

find_package(OpenCV 2)
if(OpenCV_FOUND)
    add_definitions(-DHAS_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${OpenCV_LIBS})
endif()

target_link_libraries(tensorrt libnvinfer.so.2)
target_link_libraries(tensorrt nvcaffe_parser)

install(TARGETS tensorrt RUNTIME DESTINATION bin)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
