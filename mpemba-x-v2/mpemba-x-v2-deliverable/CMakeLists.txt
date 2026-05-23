cmake_minimum_required(VERSION 3.16)
project(mpemba_x_v2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -funroll-loops")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

function(add_mpemba_module name src)
    add_executable(${name} ${src})
    target_link_libraries(${name}
        PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_compile_options(${name}
        PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
endfunction()

# Existing modules (refactored to use config files)
add_mpemba_module(mpemba_markovian          modules/markovian/main.cpp)
add_mpemba_module(mpemba_klich_raz          modules/klich_raz/main.cpp)
add_mpemba_module(mpemba_granular_analytic  modules/granular/main_analytic.cpp)
add_mpemba_module(mpemba_granular_dsmc      modules/granular/main_dsmc.cpp)
add_mpemba_module(mpemba_langevin           modules/langevin/main.cpp)

# New modules (v2)
add_mpemba_module(mpemba_langevin_inverse   modules/langevin_inverse/main.cpp)
add_mpemba_module(mpemba_water_fourier      modules/water_fourier/main.cpp)
add_mpemba_module(mpemba_water_md           modules/water_md/main.cpp)
add_mpemba_module(mpemba_quantum_lindblad   modules/quantum_lindblad/main.cpp)
add_mpemba_module(mpemba_thermomajorization modules/thermomajorization/main.cpp)

message(STATUS "Mpemba-X v2 configured")
message(STATUS "  CXX:    ${CMAKE_CXX_COMPILER}")
message(STATUS "  MPI:    ${MPI_CXX_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
