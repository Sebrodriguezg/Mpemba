CXX = g++
MPICXX = mpicxx
NVCC = nvcc

CXXFLAGS = -std=c++17 -O3
OMPFLAGS = -fopenmp
CUDAFLAGS = -O3

TARGETS = serial omp mpi cuda

all: $(TARGETS)

serial: serial.cpp
	$(CXX) $(CXXFLAGS) serial.cpp -o mpemba_serial

omp: omp_lbm.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) omp_lbm.cpp -o mpemba_omp

mpi: mpi_lbm.cpp
	$(MPICXX) $(CXXFLAGS) mpi_lbm.cpp -o mpemba_mpi

cuda: cuda_lbm.cu
	$(NVCC) $(CUDAFLAGS) cuda_lbm.cu -o mpemba_cuda

clean:
	rm -f $(TARGETS) mpemba_serial mpemba_omp mpemba_mpi mpemba_cuda
