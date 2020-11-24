cmake_minimum_required(VERSION 3.15)

project(
    NumberDetector
    DESCRIPTION "Study project"
    LANGUAGES C CXX
    VERSION 0.0.0
)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_subdirectory(src)

message(STATUS "bindir = " ${PROJECT_BINARY_DIR})