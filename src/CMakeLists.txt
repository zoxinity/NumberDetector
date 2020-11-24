cmake_minimum_required(VERSION 3.15)

project(NumberDetectorProj)

set(SOURCE_FILES NumberDetector.cpp)
add_executable(NumberDetector ${SOURCE_FILES})

target_link_libraries(
    NumberDetector
        PRIVATE
            ${OpenCV_LIBS}
            DrawText
)

target_include_directories(
    NumberDetector
        PRIVATE
            ${PROJECT_SOURCE_DIR}/drawText
)

add_subdirectory(drawText)