add_library(celldiv_lib STATIC)

# We need this directory, and users of our library will need it too
target_include_directories(celldiv_lib PUBLIC ../inc)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc")

target_sources(celldiv_lib PRIVATE
  centermass.cu
  postscriptinit.cu
  PressureKernels.cu
  propagatebound.cu
  propagate.cu
  volume.cu
  BondKernels.cu)

set_source_files_properties(
  centermass.cu
  postscriptinit.cu
  PressureKernels.cu
  propagatebound.cu
  propagate.cu
  volume.cu
  BondKernels.cu PROPERTIES LANGUAGE HIP)
add_subdirectory(utils)

add_executable(cellDiv GPUbounce.cu)
set_source_files_properties( GPUbounce.cu PROPERTIES LANGUAGE HIP)
target_link_libraries(cellDiv PRIVATE celldiv_lib utils_lib hip::hiprand m ${MPI_C_LIBRARIES})




add_executable(cellDiv_gpuawarempi GPUbounce-CudaAwareMPI.cu)
set_source_files_properties( GPUbounce-CudaAwareMPI.cu PROPERTIES LANGUAGE HIP)
target_link_libraries(cellDiv_gpuawarempi PRIVATE celldiv_lib utils_lib hip::hiprand m ${MPI_C_LIBRARIES})
