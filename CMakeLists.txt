# collect source files
cmake_minimum_required(VERSION 3.4)

find_package(CUDA REQUIRED)

FIND_LIBRARY(ACCELERATE_LIBRARY Accelerate)

set(MALLOC_INCLUDE "/usr/include/malloc")
set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g -O3 -msse4 -ljpeg -lpng -fopenmp" )
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin /usr/bin/clang -O3 -arch=sm_20 -Xcompiler -Wall" )

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
file( GLOB  cpp  *.cpp *.c*)

include_directories(${MALLOC_INCLUDE})
CUDA_ADD_EXECUTABLE(epic_flow ${hdr} ${cpp} ${cu})
target_link_libraries(epic_flow ${ACCELERATE_LIBRARY})