cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(YoloOnnxC++Tutorial)

set(ONNXRUNTIME_ROOT "/home/alex/Cpp-Libs/onnxruntime-linux-x64-gpu-1.22.0")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

find_package(OpenCV REQUIRED)

add_executable(object_detection object_detection.cpp letter_box.h)

target_link_libraries(object_detection PUBLIC onnxruntime ${OpenCV_LIBS} ${CUDA_LIBRARIES})

set_property(TARGET object_detection PROPERTY CXX_STANDARD 20)
set_target_properties(object_detection PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)