# Detect OS
ifeq ($(OS),Windows_NT)
    EXEC_EXT = .exe
    OBJ_EXT = .obj
    RM_CMD = rm -f
    MKDIR_CMD = mkdir
    PATH_SEP = \\
else
    EXEC_EXT = 
    OBJ_EXT = .o
    RM_CMD = rm -f
    MKDIR_CMD = mkdir -p
    PATH_SEP = /
endif

# Compiler and flags
NVCC = nvcc
CFLAGS = -std=c++17 --extended-lambda

# Directories
INCLUDE_DIR = include
KERNELS_DIR = kernels
BUILD_DIR = build

# Automatically detect all .cu files
KERNEL_SOURCES = $(wildcard $(KERNELS_DIR)/*.cu)
ROOT_SOURCES = bench.cu
CUDA_SOURCES = $(KERNEL_SOURCES) $(ROOT_SOURCES)

# Generate object file names
CUDA_OBJECTS = $(patsubst $(KERNELS_DIR)/%.cu,$(BUILD_DIR)/%$(OBJ_EXT),$(KERNEL_SOURCES)) $(BUILD_DIR)/bench$(OBJ_EXT)

# Output executable
OUTPUT = softmax$(EXEC_EXT)

# Rules
all: $(OUTPUT)

$(OUTPUT): $(CUDA_OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Rules for compiling kernels directory .cu files
$(BUILD_DIR)/%$(OBJ_EXT): $(KERNELS_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Rule for compiling root directory .cu files
$(BUILD_DIR)/bench$(OBJ_EXT): bench.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Create the build directory
$(BUILD_DIR):
	$(MKDIR_CMD) $(BUILD_DIR)

# Clean rule
clean:
ifeq ($(OS),Windows_NT)
	if exist $(BUILD_DIR) $(RM_CMD) $(BUILD_DIR)\*$(OBJ_EXT)
	if exist $(OUTPUT) $(RM_CMD) $(OUTPUT)
	if exist *.exp $(RM_CMD) *.exp
	if exist *.lib $(RM_CMD) *.lib
else
	$(RM_CMD) $(BUILD_DIR)/*$(OBJ_EXT) $(OUTPUT)
endif

.PHONY: all clean