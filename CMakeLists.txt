cmake_minimum_required(VERSION 3.27)
project(vision_processor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -Wall -ftree-vectorize -ffast-math")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_VERBOSE_MAKEFILE 1)
add_compile_options(-fopt-info-vec-optimized-missed)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenCL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)

find_package(Protobuf REQUIRED)
file(GLOB PROTO_DEFS proto/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_DEFS})

find_package(Spinnaker)

file(GLOB SRC src/*.cpp src/*.c)
add_executable(${PROJECT_NAME} ${SRC} ${PROTO_SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE src ${OpenCV_INCLUDE_DIRS} ${SPINNAKER_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} PkgConfig::LIBAV ${SPINNAKER_LIBS} OpenCL::OpenCL ${Protobuf_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300)

install(TARGETS ${PROJECT_NAME} DESTINATION /usr/local/bin)
