REPO_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
include $(REPO_ROOT)/Makefile.common

all: gpu_forwarder

gpu_forwarder.o: gpu_forwarder.c gpu_forwarder.h
	$(CC) $(CFLAGS) -c -o $@ $<

gpu_forwarder_kernel.o: gpu_forwarder_kernel.cu gpu_forwarder.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

gpu_forwarder: gpu_forwarder.o gpu_forwarder_kernel.o
	$(NVCC) \
	    -gencode arch=compute_75,code=sm_75 \
	    -gencode arch=compute_80,code=sm_80 \
	    -gencode arch=compute_86,code=sm_86 \
	    -gencode arch=compute_90,code=sm_90 \
	    -o $@ $^ $(GPU_LDFLAGS)

clean:
	rm -f gpu_forwarder *.o
