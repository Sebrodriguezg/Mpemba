cmake_minimum_required(VERSION 3.16)
project(qmpe_hpc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qmpe_hpc qmpe_hpc.cpp)
target_link_libraries(qmpe_hpc PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(qmpe_hpc PRIVATE -Wall -Wextra)

message(STATUS "qmpe_hpc configurado")
message(STATUS "  CXX:    ${CMAKE_CXX_COMPILER}")
message(STATUS "  MPI:    ${MPI_CXX_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
