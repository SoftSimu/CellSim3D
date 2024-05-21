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

add_subdirectory(utils)

add_executable(cellDiv GPUbounce.cu)
set_property(TARGET cellDiv PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET celldiv_lib cellDiv PROPERTY CUDA_ARCHITECTURES native)
target_link_libraries(cellDiv PRIVATE celldiv_lib utils_lib ${CUDA_curand_LIBRARY} ${CUDA_LIBRARIES} m)
