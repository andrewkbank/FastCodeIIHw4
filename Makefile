# Compiler and Flags
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -fopenmp -arch=sm_75 -std=c++11

# Executable Names
ORIGINAL_EXE = main_original.x
OPTIMIZED_EXE = main_optimized.x

# Default target: build both
all: $(ORIGINAL_EXE) $(OPTIMIZED_EXE)

# Build the original/legacy version
# Note: Keeping -x cu because the original main.cpp contains CUDA kernels
$(ORIGINAL_EXE): main.cpp
	$(NVCC) -x cu main.cpp $(NVCC_FLAGS) -o $(ORIGINAL_EXE)

# Build the new optimized version
$(OPTIMIZED_EXE): main.cu
	$(NVCC) main.cu $(NVCC_FLAGS) -o $(OPTIMIZED_EXE)

# Run both for comparison
run: all
	@echo "--- Running Original Implementation ---"
	./$(ORIGINAL_EXE) 128
	@echo ""
	@echo "--- Running Optimized (Pipelined) Implementation ---"
	./$(OPTIMIZED_EXE) 128

# Clean up binaries
clean:
	rm -f $(ORIGINAL_EXE) $(OPTIMIZED_EXE)