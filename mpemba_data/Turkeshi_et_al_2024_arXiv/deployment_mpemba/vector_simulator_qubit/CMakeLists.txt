cmake_minimum_required(VERSION 3.5)

project(quantum_mpemba_ruc VERSION 1.0
        DESCRIPTION "Quantum Mpemba Effect in Random Unitary Circuits"
        LANGUAGES Fortran)

set(CMAKE_Fortran_COMPILER gfortran)

if(APPLE)
  # Use Accelerate on macOS
  set(dialect "-ffree-form -std=f2018 -fimplicit-none -O3")
  set(link_libs "-framework" "Accelerate")
else()
  # Original MKL path for Linux or Intel systems
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(dialect "-ffree-form -std=f2018 -fimplicit-none -O3 -L /home/xhek.turkeshi/intel/oneapi/mkl/2021.4.0/lib/intel64 -lmkl_rt")
  endif()
endif()

set(bounds "-fbounds-check")
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${bounds}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${dialect}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FVERSION}")

include_directories(src)

# Output folders
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)

# Sources without main
set(SOURCE_FILES_NO_MAIN
  src/mod_utils.f90
  src/mod_basis.f90
  src/mod_haar.f90
)

add_library(symmetric_circuits ${SOURCE_FILES_NO_MAIN})
set_target_properties(symmetric_circuits PROPERTIES PREFIX "")
target_include_directories(symmetric_circuits PUBLIC src)
target_compile_options(symmetric_circuits PRIVATE -O3)
target_link_options(symmetric_circuits PRIVATE -O3 ${link_libs})

# Main program(s)
file(GLOB APP_SOURCES src/main.f90)
foreach(source ${APP_SOURCES})
  get_filename_component(exec ${source} NAME_WE)
  add_executable(${exec} ${source})
  target_link_libraries(${exec} symmetric_circuits)
  target_compile_options(${exec} PRIVATE -O3)
  target_link_options(${exec} PRIVATE -O3 ${link_libs})
endforeach()