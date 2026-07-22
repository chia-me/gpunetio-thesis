# Makefile per gpu_bridge
# Eredita flag comuni (DOCA_INC, CUDA_HOME, NVFLAGS, GPU_LDFLAGS) da Makefile.common.
# Uso:
#   make              → compila gpu_bridge
#   make clean        → rimuove file compilati

REPO_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(REPO_ROOT)/Makefile.common

TARGET  := gpu_bridge
SRCS_C  := gpu_bridge.c
SRCS_CU := gpu_bridge_kernel.cu
OBJS    := $(SRCS_C:.c=.o) $(SRCS_CU:.cu=.o)

.PHONY: all clean

all: $(TARGET)

# Linka con nvcc perché libdoca_gpunetio_device.a è un archivio device code
$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(GPU_LDFLAGS)

gpu_bridge.o: gpu_bridge.c gpu_bridge.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu_bridge_kernel.o: gpu_bridge_kernel.cu gpu_bridge.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
