cmake_minimum_required(VERSION 3.22)
project(vision_processor)

option(WITH_SPINNAKER "Build with Spinnaker support (if library available)" ON)
option(WITH_MVIMPACT "Build with mvIMPACT/Bluefox support (if library available)" ON)

# Adapted from https://stackoverflow.com/a/48832234 CC BY-SA 4.0 by einpoklum
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -Wall -ftree-vectorize -ffast-math")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio bgsegm)
find_package(PkgConfig REQUIRED)
find_package(Eigen3 REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)
find_package(Protobuf REQUIRED)
find_package(OpenCL REQUIRED)

find_package(Spinnaker)
find_package(mvIMPACT)

add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=300)
link_libraries(${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} PkgConfig::LIBAV ${SPINNAKER_LIBS} OpenCL::OpenCL ${Protobuf_LIBRARIES} ${mvIMPACT_LIBS})
include_directories(src ${YAML_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SPINNAKER_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} Eigen3::Eigen ${mvIMPACT_INCLUDE_DIRS})

file(GLOB PROTO_FILES proto/*.proto)
file(GLOB CL_KERNELS RELATIVE "kernel/" kernel/*.cl)
file(GLOB_RECURSE SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry_benchmark.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/blob_benchmark.cpp")

add_executable(${PROJECT_NAME} ${SRC} "src/main.cpp")
add_executable("geometry_benchmark" ${SRC} "src/geometry_benchmark.cpp")
add_executable("blob_benchmark" ${SRC} "src/blob_benchmark.cpp")

# Adapted from https://stackoverflow.com/a/47801116 CC BY-SA 3.0 by Martin R.
foreach(input_file IN LISTS CL_KERNELS)
    file(READ "kernel/${input_file}" content)
    set(content "R\"PrepForInclude(${content}\n)PrepForInclude\"")
    file(WRITE "src/kernel/${input_file}" "${content}")
    unset(content)
endforeach()

protobuf_generate(TARGET ${PROJECT_NAME} LANGUAGE cpp PROTOC_OUT_DIR "src" PROTOS ${PROTO_FILES})
protobuf_generate(TARGET "geometry_benchmark" LANGUAGE cpp PROTOC_OUT_DIR "src" PROTOS ${PROTO_FILES})
protobuf_generate(TARGET "blob_benchmark" LANGUAGE cpp PROTOC_OUT_DIR "src" PROTOS ${PROTO_FILES})
protobuf_generate(TARGET ${PROJECT_NAME} LANGUAGE python PROTOC_OUT_DIR "python" PROTOS ${PROTO_FILES} PROTOC_OPTIONS --pyi_out ":${CMAKE_CURRENT_SOURCE_DIR}/python")

#file(GLOB_RECURSE HEADERS "src/*.h")
#target_precompile_headers(${PROJECT_NAME} PRIVATE ${HEADERS})

install(TARGETS ${PROJECT_NAME} DESTINATION /usr/local/bin)
