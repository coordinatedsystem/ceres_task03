cmake_minimum_required(VERSION 3.10)
project(videoCurveFit)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS highgui imgcodecs videoio)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 未找到，请确保已正确安装。")
endif()
find_package(Ceres REQUIRED)
if(CERES_FOUND)
    message(STATUS "Found Ceres: ${CERES_INCLUDE_DIRS}, ${CERES_LIBRARIES}")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Libraries: ${OpenCV_LIBS}")
message(STATUS "Include dirs: ${OpenCV_INCLUDE_DIRS}")

add_executable(videoCurveFit src/main.cpp)

target_link_libraries(videoCurveFit ${OpenCV_LIBS})
target_link_libraries(videoCurveFit
    ${CERES_LIBRARIES}
)
target_include_directories(videoCurveFit PRIVATE
    ${CERES_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
)

set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/output)

file(MAKE_DIRECTORY ${OUTPUT_DIR})
add_definitions(-DOUTPUT_DIR="${OUTPUT_DIR}")

target_compile_options(videoCurveFit PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -O2>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)