# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 -allow-unsupported-compiler -lcublas

# Directories
INCLUDE_DIR = include
KERNELS_DIR = kernels
BUILD_DIR = build

# Automatically detect all .cu files
KERNEL_SOURCES = $(wildcard $(KERNELS_DIR)/*.cu)
ROOT_SOURCES = src.cu
CUDA_SOURCES = $(KERNEL_SOURCES) $(ROOT_SOURCES)

# Generate object file names
CUDA_OBJECTS = $(patsubst $(KERNELS_DIR)/%.cu,$(BUILD_DIR)/%.obj,$(KERNEL_SOURCES)) $(BUILD_DIR)/src.obj

# Output executable
OUTPUT = matvec.exe

# Rules
all: $(OUTPUT)

$(OUTPUT): $(CUDA_OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Rules for compiling kernels directory .cu files
$(BUILD_DIR)/%.obj: $(KERNELS_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Rule for compiling root directory .cu files
$(BUILD_DIR)/src.obj: src.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Create the build directory
$(BUILD_DIR):
	mkdir $(BUILD_DIR)

# Clean rule
clean:
	rm $(BUILD_DIR)/*.obj *.exe *.exp *.lib

.PHONY: all clean
