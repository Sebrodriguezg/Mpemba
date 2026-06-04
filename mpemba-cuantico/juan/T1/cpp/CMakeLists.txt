cmake_minimum_required(VERSION 3.16)
project(qmpe_t1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -funroll-loops")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# LAPACK para el pequeno problema de autovalores del Hessenberg (zgeev).
# En macOS se usa el framework Accelerate; en Linux, find_package(LAPACK).
if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    set(LAPACK_LIBS ${ACCELERATE_LIB})
else()
    find_package(LAPACK REQUIRED)
    set(LAPACK_LIBS ${LAPACK_LIBRARIES})
endif()

add_executable(arnoldi_lindblad_mpi arnoldi_lindblad_mpi.cpp)
target_link_libraries(arnoldi_lindblad_mpi
    PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${LAPACK_LIBS})
target_compile_options(arnoldi_lindblad_mpi
    PRIVATE -Wall -Wextra -Wno-unused-parameter)

message(STATUS "qmpe_t1 configured")
message(STATUS "  CXX:    ${CMAKE_CXX_COMPILER}")
message(STATUS "  MPI:    ${MPI_CXX_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  LAPACK: ${LAPACK_LIBS}")
