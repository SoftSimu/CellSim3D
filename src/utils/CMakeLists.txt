
add_library(utils_lib STATIC)
message("${CMAKE_SOURCE_DIR}")
target_include_directories(utils_lib PUBLIC "${CMAKE_SOURCE_DIR}/inc")

target_sources(utils_lib PRIVATE jsoncpp.cpp) 
