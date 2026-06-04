cmake_minimum_required(VERSION 3.16)
project(qmpe_t3a CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtraj_mpi qtraj_mpi.cpp)
target_link_libraries(qtraj_mpi PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(qtraj_mpi PRIVATE -Wall -Wextra -Wno-unused-parameter)

message(STATUS "qmpe_t3a configured (MPI ${MPI_CXX_VERSION}, OpenMP ${OpenMP_CXX_VERSION})")
