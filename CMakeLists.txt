cmake_minimum_required(VERSION 2.8)

project(demo)

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(${CMAKE_PROJECT_NAME} main.cpp src/ocv_runner.cpp src/ie_runner.cpp)
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_range_for)

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBS}
  ${InferenceEngine_LIBRARIES}
)
