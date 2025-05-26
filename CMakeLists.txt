cmake_minimum_required(VERSION 3.22.1)
project(gm_cuda CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_STANDARD 20)

add_executable(gm_cuda main.cu)

find_package(OpenCV CONFIG REQUIRED)

target_include_directories(gm_cuda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gm_cuda PRIVATE ${OpenCV_LIBS})

set_target_properties(gm_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET gm_cuda PROPERTY CUDA_ARCHITECTURES 89)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
