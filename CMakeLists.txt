cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(CELLDIV LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

add_subdirectory(src)



