/*
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

/**
 * @file doca_gpunetio_dev_eth_common.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCA_GPUNETIO_DEV_ETH_COMMON DOCA GPUNetIO Device - Ethernet Common
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_ETH_COMMON_H
#define DOCA_GPUNETIO_DEVICE_ETH_COMMON_H

#include <cuda.h>
#include <cuda/atomic>
#include <doca_gpunetio_eth_def.h>

#if CUDA_VERSION >= 12020
#define DOCA_GPUNETIO_ETH_HAS_STORE_RELAXED_MMIO 1
#endif

#if CUDA_VERSION >= 12080 && __CUDA_ARCH__ >= 900
#define DOCA_GPUNETIO_ETH_HAS_FENCE_ACQUIRE_RELEASE_PTX 1
#endif

/**
 * Macro to temporarily cast a variable to volatile.
 */
#define DOCA_GPUNETIO_ETH_VOLATILE(x) (*(volatile typeof(x) *)&(x))

/**
 * Macro to write volatile
 */
#define DOCA_GPUNETIO_ETH_WRITE_ONCE(x, v) (DOCA_GPUNETIO_ETH_VOLATILE(x) = (v))

/**
 * @typedef doca_gpu_dev_eth_ticket_t
 * @brief Ticket type used in one-sided APIs.
 */
typedef uint64_t doca_gpu_dev_eth_ticket_t;

/*********************************************************************************************************************
 * Structures
 *********************************************************************************************************************/

/**
 * Describes GPUNetIO dev WQE crtl segment.
 */
struct doca_gpu_dev_eth_wqe_ctrl_seg {
	__be32 opmod_idx_opcode; /**< opcode + wqe idx */
	__be32 qpn_ds;		 /**< qp number */
	union {
		struct {
			uint8_t signature; /**< signature */
			uint8_t rsvd[2];   /**< reserved */
			uint8_t fm_ce_se;  /**< flags */
		};
		struct {
			__be32 flags; /**< all flags in or */
		};
	};

	__be32 imm; /**< immediate */
} __attribute__((__aligned__(8)));

/*********************************************************************************************************************
 * Utility functions
 *********************************************************************************************************************/

/**
 * @brief Queries the global timer
 *
 * @return The value of the global timer
 */
__device__ static inline uint64_t doca_gpu_dev_eth_query_globaltimer()
{
	uint64_t ret;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
	return ret;
}

/**
 * @brief Swap a 64bits variable
 *
 * @param [in] x - Value to be swapped
 * @return The swapped value
 */
__device__ static inline uint64_t doca_gpu_dev_eth_bswap64(uint64_t x)
{
	uint64_t ret;
	asm volatile("{\n\t"
		     ".reg .b32 mask;\n\t"
		     ".reg .b32 ign;\n\t"
		     ".reg .b32 lo;\n\t"
		     ".reg .b32 hi;\n\t"
		     ".reg .b32 new_lo;\n\t"
		     ".reg .b32 new_hi;\n\t"
		     "mov.b32 mask, 0x0123;\n\t"
		     "mov.b64 {lo,hi}, %1;\n\t"
		     "prmt.b32 new_hi, lo, ign, mask;\n\t"
		     "prmt.b32 new_lo, hi, ign, mask;\n\t"
		     "mov.b64 %0, {new_lo,new_hi};\n\t"
		     "}"
		     : "=l"(ret)
		     : "l"(x));
	return ret;
}

/**
 * @brief Swap a 32bits variable
 *
 * @param [in] x - Value to be swapped
 * @return The swapped value
 */
__device__ static inline uint32_t doca_gpu_dev_eth_bswap32(uint32_t x)
{
	uint32_t ret;
	asm volatile("{\n\t"
		     ".reg .b32 mask;\n\t"
		     ".reg .b32 ign;\n\t"
		     "mov.b32 mask, 0x0123;\n\t"
		     "prmt.b32 %0, %1, ign, mask;\n\t"
		     "}"
		     : "=r"(ret)
		     : "r"(x));
	return ret;
}

/**
 * @brief Store a 16Bytes (2*64bit) WQE segment
 *
 * @param [in] ptr - store destination memory value
 * @param [in] val - store source memory value
 * @return
 */
__device__ static inline void doca_gpu_dev_eth_store_wqe_seg(uint64_t *ptr, uint64_t *val)
{
	asm volatile("st.weak.cs.v2.b64 [%0], {%1, %2};" : : "l"(ptr), "l"(val[0]), "l"(val[1]));
}

/**
 * @brief Return CUDA thread lane ID in the current warp
 *
 * @return The lane ID
 */
__device__ static inline unsigned int doca_gpu_dev_eth_get_lane_id()
{
	unsigned int ret;
	asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret));
	return ret;
}

/**
 * @brief Store relaxed 64bits using .mmio CUDA PTX operation
 *
 * @param [in] ptr - store destination memory value
 * @param [in] val - value to store
 *
 * @return
 */
#ifdef DOCA_GPUNETIO_ETH_HAS_STORE_RELAXED_MMIO
__device__ static inline void doca_gpu_dev_eth_store_relaxed_mmio(uint64_t *ptr, uint64_t val)
{
	asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}
#endif

/**
 * @brief Load relaxed 8bits using CUDA PTX
 *
 * @param [in] ptr - Memory to load
 *
 * @return
 */
__device__ static inline uint8_t doca_gpu_dev_eth_load_relaxed_sys_global(uint8_t *ptr)
{
	uint16_t ret;
	asm volatile("ld.relaxed.sys.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
	return (uint8_t)ret;
}

/**
 * @brief Load relaxed 16bits using CUDA PTX
 *
 * @param [in] ptr - Memory to load
 *
 * @return
 */
__device__ static inline uint16_t doca_gpu_dev_eth_load_relaxed_sys_global(uint16_t *ptr)
{
	uint16_t ret;
	asm volatile("ld.relaxed.sys.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
	return (uint16_t)ret;
}


/**
 * @brief Load relaxed 64bits using CUDA PTX
 *
 * @param [in] ptr - Memory to load
 *
 * @return
 */
__device__ static inline uint64_t doca_gpu_dev_eth_load_relaxed_sys_global(uint64_t *ptr)
{
	uint64_t ret;
	asm volatile("ld.relaxed.sys.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
	return ret;
}

/**
 * @brief Fetch max CUDA C++ operation for different resource scopes
 *
 * @param [in] ptr - memory used in the fetch_max operation
 * @param [in] val - value for the fetch_max
 *
 * @return Previous value in memory
 */
template <typename T, enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode, bool need_fence_acquire = false>
__device__ static inline T doca_gpu_dev_eth_atomic_max(T *ptr, T val)
{
	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE) {
		T old_val = *ptr;
		*ptr = max(old_val, val);
		return old_val;
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA) {
		cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
		return ptr_aref.fetch_max(val,
					  need_fence_acquire ? cuda::std::memory_order_acquire :
							       cuda::std::memory_order_relaxed);
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU) {
		cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
		return ptr_aref.fetch_max(val,
					  need_fence_acquire ? cuda::std::memory_order_acquire :
							       cuda::std::memory_order_relaxed);
	}
	return 0;
}

/**
 * @brief Fetch add CUDA C++ operation for different resource scopes
 *
 * @param [in] ptr - memory used in the fetch_add operation
 * @param [in] val - value for the fetch_add
 *
 * @return Previous value in memory
 */
template <typename T, enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode>
__device__ static inline T doca_gpu_dev_eth_atomic_add(T *ptr, T val)
{
	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE) {
		T old_val = *ptr;
		*ptr = old_val + val;
		return old_val;
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA) {
		cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
		return ptr_aref.fetch_add(val, cuda::std::memory_order_relaxed);
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU) {
		cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
		return ptr_aref.fetch_add(val, cuda::std::memory_order_relaxed);
	}
	return 0;
}

/**
 * @brief Volatile read for different resource scopes
 *
 * @param [in] ptr - memory to read
 *
 * @return Read value
 */
template <typename T, enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode>
__device__ static inline T doca_gpu_dev_eth_atomic_read(T *ptr)
{
	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE)
		return *ptr;
	else
		return DOCA_GPUNETIO_ETH_VOLATILE(*ptr);
}

/**
 * @brief CUDA PTX fence acquire for different scopes
 *
 * @return
 */
template <enum doca_gpu_dev_eth_sync_scope sync_scope>
__device__ static inline void doca_gpu_dev_eth_fence_acquire()
{
#ifdef DOCA_GPUNETIO_ETH_HAS_FENCE_ACQUIRE_RELEASE_PTX
	if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA)
		asm volatile("fence.acquire.cta;");
	else if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU)
		asm volatile("fence.acquire.gpu;");
	else
		asm volatile("fence.acquire.sys;");
#else
	// fence.release is not available in PTX
	if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA)
		asm volatile ("membar.cta;" ::: "memory");
	else if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU)
		asm volatile ("membar.gl;" ::: "memory");
	else
		asm volatile ("membar.sys;" ::: "memory");
#endif
}

/**
 * @brief CUDA PTX fence release for different scopes
 *
 * @return
 */
template <enum doca_gpu_dev_eth_sync_scope sync_scope>
__device__ static inline void doca_gpu_dev_eth_fence_release()
{
#ifdef DOCA_GPUNETIO_ETH_HAS_FENCE_ACQUIRE_RELEASE_PTX
	if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA)
		asm volatile("fence.release.cta;");
	else if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU)
		asm volatile("fence.release.gpu;");
	else
		asm volatile("fence.release.sys;");
#else
	// fence.release is not available in PTX
	if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA)
		asm volatile ("membar.cta;" ::: "memory");
	else if (sync_scope == DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU)
		asm volatile ("membar.gl;" ::: "memory");
	else
		asm volatile ("membar.sys;" ::: "memory");
#endif
}

/**
 * @brief Lock a resource with CUDA atomicCAS
 *
 * @param [in] lock - Pointer to the lock
 *
 * @return
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode>
__device__ static inline void doca_gpu_dev_eth_lock(int *lock)
{
	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE) {
		*lock = 1;
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA) {
		while (atomicCAS_block(lock, 0, 1) != 0)
			continue;
		doca_gpu_dev_eth_fence_acquire<DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>();
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU) {
		while (atomicCAS(lock, 0, 1) != 0)
			continue;
		doca_gpu_dev_eth_fence_acquire<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
	}
}

/**
 * @brief Unlock a resource with CUDA atomic_ref store
 *
 * @param [in] lock - Pointer to the lock
 *
 * @return
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode>
__device__ static inline void doca_gpu_dev_eth_unlock(int *lock)
{
	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE) {
		*lock = 0;
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA) {
		cuda::atomic_ref<int, cuda::thread_scope_block> lock_aref(*lock);
		lock_aref.store(0, cuda::std::memory_order_release);
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU) {
		cuda::atomic_ref<int, cuda::thread_scope_device> lock_aref(*lock);
		lock_aref.store(0, cuda::std::memory_order_release);
	}
}

/**
 * @brief Print error CQE values
 *
 * @param [in] cqe64 - erroneous cqe
 *
 * @return
 */
__device__ static inline void doca_gpu_dev_eth_print_cqe_err(volatile struct mlx5_cqe64 *cqe64)
{
	struct mlx5_err_cqe_ex *err_cqe = (struct mlx5_err_cqe_ex *)cqe64;

	printf("got completion with err: "
	       "syndrome=%#x, vendor_err_synd=%#x, "
	       "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u wqe_qpn=%x\n",
	       err_cqe->syndrome,
	       err_cqe->vendor_err_synd,
	       err_cqe->hw_err_synd,
	       err_cqe->hw_synd_type,
	       DOCA_GPUNETIO_ETH_BSWAP16(err_cqe->wqe_counter),
	       doca_gpu_dev_eth_bswap32(err_cqe->s_wqe_opcode_qpn));
}

/**
 * @brief Prepare the DBREC (Doorbell Record) for Txq
 *
 * @param [in] prod_index - Producer index to store in DBREC
 * @return DBREC value
 */
__device__ static inline __be32 doca_gpu_dev_eth_prepare_dbr(uint32_t prod_index)
{
	__be32 dbrec_val;

	// This is equivalent to HTOBE32(dbrec_head & 0xffff);
	asm volatile("{\n\t"
		     ".reg .b32 mask1;\n\t"
		     ".reg .b32 dbrec_head_16b;\n\t"
		     ".reg .b32 ign;\n\t"
		     ".reg .b32 mask2;\n\t"
		     "mov.b32 mask1, 0xffff;\n\t"
		     "mov.b32 mask2, 0x123;\n\t"
		     "and.b32 dbrec_head_16b, %1, mask1;\n\t"
		     "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t"
		     "}"
		     : "=r"(dbrec_val)
		     : "r"(prod_index));

	return dbrec_val;
}

/**
 * @brief Prepare the DBREC (Doorbell Record) for CQ
 *
 * @param [in] prod_index - Producer index
 * @return DBREC value
 */
__device__ static inline __be32 doca_gpu_dev_eth_prepare_cq_dbr(uint32_t prod_index)
{
	return doca_gpu_dev_eth_bswap32(prod_index & DOCA_GPUNETIO_ETH_CQE_CI_MASK);
}

#endif /* DOCA_GPUNETIO_DEVICE_ETH_COMMON_H */

/** @} */
