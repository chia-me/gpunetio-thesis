# Makefile per gpu_router
# Eredita flag comuni (DOCA_INC, CUDA_HOME, NVFLAGS, GPU_LDFLAGS) da Makefile.common.
# Uso:
#   make              → compila gpu_router
#   make clean        → rimuove file compilati

REPO_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(REPO_ROOT)/Makefile.common

TARGET  := gpu_router
SRCS_C  := gpu_router.c
SRCS_CU := gpu_router_kernel.cu
OBJS    := $(SRCS_C:.c=.o) $(SRCS_CU:.cu=.o)

.PHONY: all clean

all: $(TARGET)

# Linka con nvcc perché libdoca_gpunetio_device.a è un archivio device code
$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(GPU_LDFLAGS)

gpu_router.o: gpu_router.c gpu_router.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu_router_kernel.o: gpu_router_kernel.cu gpu_router.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
