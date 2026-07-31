# Makefile per gpu_bridge_rss (v2 multi-coda RSS)
# Eredita flag comuni (DOCA_INC, CUDA_HOME, NVFLAGS, GPU_LDFLAGS) da Makefile.common.
# Uso:
#   make              → compila gpu_bridge_rss
#   make clean        → rimuove file compilati

REPO_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(REPO_ROOT)/Makefile.common

TARGET  := gpu_bridge_rss
SRCS_C  := gpu_bridge_rss.c
SRCS_CU := gpu_bridge_rss_kernel.cu
OBJS    := $(SRCS_C:.c=.o) $(SRCS_CU:.cu=.o)

.PHONY: all clean

all: $(TARGET)

# Linka con nvcc perché libdoca_gpunetio_device.a è un archivio device code
$(TARGET): $(OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $^ $(GPU_LDFLAGS)

gpu_bridge_rss.o: gpu_bridge_rss.c gpu_bridge_rss.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu_bridge_rss_kernel.o: gpu_bridge_rss_kernel.cu gpu_bridge_rss.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
