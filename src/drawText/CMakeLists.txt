cmake_minimum_required(VERSION 3.15)

project(DrawTextProj)

set(SOURCE_FILES
    DrawText.hpp
    DrawText.cpp
)
add_library(DrawText STATIC ${SOURCE_FILES})
target_link_libraries(DrawText PRIVATE ${OpenCV_LIBS})

target_include_directories(
    DrawText
        PRIVATE
            ${OpenCV_INCLUDE_DIRS}
)