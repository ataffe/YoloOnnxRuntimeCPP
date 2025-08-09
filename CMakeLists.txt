cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(YoloOnnxC++Tutorial)

include_directories(/usr/local/include)
include_directories(/usr/local/include/opencv4)
include_directories(/usr/local/include/onnxruntime/include)
link_directories(/usr/local/include/onnxruntime/lib)

find_package(OpenCV REQUIRED)

add_executable(object_detection object_detection.cpp
        object_detection.cpp
)

target_link_libraries(object_detection PUBLIC ${OpenCV_LIBS})
target_link_libraries(object_detection PUBLIC onnxruntime)

set_property(TARGET object_detection PROPERTY CXX_STANDARD 20)