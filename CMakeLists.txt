cmake_minimum_required(VERSION 3.22)
project(vision_processor)

option(WITH_SPINNAKER "Build with Spinnaker support (if library available)" ON)

# Adapted from https://stackoverflow.com/a/48832234 CC BY-SA 4.0 by einpoklum
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

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
find_package(Protobuf REQUIRED)
find_package(OpenCL REQUIRED)

find_package(Spinnaker)

file(GLOB PROTO_FILES proto/*.proto)
file(GLOB CL_KERNELS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/kernel/" kernel/*.cl)
file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE src ${YAML_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SPINNAKER_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} PkgConfig::LIBAV ${SPINNAKER_LIBS} OpenCL::OpenCL ${Protobuf_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_HPP_TARGET_OPENCL_VERSION=300)

# Adapted from https://stackoverflow.com/a/47801116 CC BY-SA 3.0 by Martin R.
foreach(input_file IN LISTS CL_KERNELS)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/kernel/${input_file}" content)
    set(content "R\"PrepForInclude(${content}\n)PrepForInclude\"")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/${input_file}" "${content}")
    unset(content)
endforeach()

protobuf_generate(TARGET ${PROJECT_NAME} LANGUAGE cpp PROTOC_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" PROTOS ${PROTO_FILES})
protobuf_generate(TARGET ${PROJECT_NAME} LANGUAGE python PROTOC_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python" PROTOS ${PROTO_FILES})

#file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
#target_precompile_headers(${PROJECT_NAME} PRIVATE ${HEADERS})

install(TARGETS ${PROJECT_NAME} DESTINATION /usr/local/bin)
