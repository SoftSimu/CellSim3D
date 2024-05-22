cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(CELLDIV LANGUAGES HIP CXX C)

find_package(MPI)
find_package(hiprand)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})


add_subdirectory(src)





