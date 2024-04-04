cmake_minimum_required(VERSION 3.22)
project(vision_processor)

option(WITH_SPINNAKER "Build with Spinnaker support (if library available" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -Wall -ftree-vectorize -ffast-math")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#set(CMAKE_VERBOSE_MAKEFILE 1)
#add_compile_options(-fopt-info-vec-optimized-missed)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio bgsegm)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)

find_package(Spinnaker)

find_package(Protobuf REQUIRED)
file(GLOB PROTO_DEFS proto/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_DEFS})

find_package(OpenCL REQUIRED)
file(GLOB CL_KERNELS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/kernel/" kernel/*.cl)
# Adapted from https://stackoverflow.com/a/47801116 CC BY-SA 3.0 by Martin R.
foreach(input_file IN LISTS CL_KERNELS)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/kernel/${input_file}" content)
    set(content "R\"PrepForInclude(${content}\n)PrepForInclude\"")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${input_file}" "${content}")
endforeach()
unset(content)

file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
add_executable(${PROJECT_NAME} ${SRC} ${PROTO_SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE src ${YAML_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SPINNAKER_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} PkgConfig::LIBAV ${SPINNAKER_LIBS} OpenCL::OpenCL ${Protobuf_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300)

file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
target_precompile_headers(${PROJECT_NAME} PRIVATE ${HEADERS})

install(TARGETS ${PROJECT_NAME} DESTINATION /usr/local/bin)
