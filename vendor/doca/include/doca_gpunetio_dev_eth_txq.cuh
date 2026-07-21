/*
 * Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * @file doca_gpunetio_dev_eth_txq.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCA_GPUNETIO_DEV_ETH_TXQ DOCA GPUNetIO Device - Ethernet TXQ
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_ETH_TXQ_H
#define DOCA_GPUNETIO_DEVICE_ETH_TXQ_H

#include <doca_gpunetio_dev_eth_common.cuh>
#include <cuda.h>
#include <cuda/atomic>
#include <doca_gpunetio_eth_def.h>

/**
 * Nanoseconds per second
 */
#define DOCA_GPUNETIO_NS_PER_S 1000000000

/**
 * WQE Enhanced Multi-Packet Send data seg number
 */
#define DOCA_GPUNETIO_ETH_EMPW_DSEG_NUM 4

/**
 * Max EMPW datasegment for a single WQE
 */
#define DOCA_GPUNETIO_ETH_EMPW_DSEG_MAX_NUM 61

/**
 * Max EMPW datasegment per WQEBB
 */
#define DOCA_GPUNETIO_ETH_EMPW_SEGMENTS_PER_WQE 4

/**
 * @typedef doca_gpu_dev_eth_txq_ticket_t
 * @brief Ticket type used in one-sided APIs.
 */
typedef uint64_t doca_gpu_dev_eth_txq_ticket_t;

/*********************************************************************************************************************
 * Structures
 *********************************************************************************************************************/

/**
 * Describes GPUNetIO dev WQE Ethernet segment.
 */
struct doca_gpu_dev_eth_txq_wqe_eth_seg {
	union {
		struct {
			__be32 swp_offs;
			uint8_t cs_flags;
			uint8_t swp_flags;
			__be16 mss;
			__be32 metadata;
			__be16 inline_hdr_sz;
			union {
				__be16 inline_data;
				__be16 vlan_tag;
			};
		};
		struct {
			__be32 offsets;
			__be32 flags;
			__be32 flow_metadata;
			__be32 inline_hdr;
		};
	};
};

/**
 * Describes GPUNetIO dev WQE Wait on time segment.
 */
struct doca_gpu_dev_eth_txq_wqe_wait_seg {
	union {
		struct {
			__be32 operation;
			__be32 lkey;
			__be32 va_high;
			__be32 va_low;
			__be64 value;
			__be64 mask;
		};
		struct {
			__be64 w0[2];
			__be64 w1[2];
		};
	};
};

/**
 * Describes GPUNetIO dev WQE.
 */
struct doca_gpu_dev_eth_txq_wqe {
	struct doca_gpu_dev_eth_wqe_ctrl_seg cseg; /**< WQE control segment */
	union {
		struct {
			struct doca_gpu_dev_eth_txq_wqe_eth_seg eseg; /**< WQE Ethernet segment */
			struct mlx5_wqe_data_seg dseg0;		      /**< WQE first data segment */
			struct mlx5_wqe_data_seg dseg1;		      /**< WQE second data segment */
		};
		struct {
			struct doca_gpu_dev_eth_txq_wqe_wait_seg wseg; /**< WQE wait on time segment */
			uint64_t padding[2];			       /**< WQE padding */
		};
	};
};

/**
 * WQE Enhanced Multi-Packet Send
 */
struct doca_gpu_dev_eth_txq_wqe_empw {
	struct mlx5_wqe_data_seg dseg[DOCA_GPUNETIO_ETH_EMPW_DSEG_NUM]; /**< 4 data segments */
};

/*********************************************************************************************************************
 * Utility functions
 *********************************************************************************************************************/

/**
 * @brief Get a pointer to the Txq QP WQE buffer at a specific index
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] wqe_idx - Index of the WQE to get
 * @return Pointer to the WQE buffer at the specified index
 */
__device__ static inline struct doca_gpu_dev_eth_txq_wqe *doca_gpu_dev_eth_txq_get_wqe_ptr(struct doca_gpu_eth_txq *txq,
											   uint16_t wqe_idx)
{
	const uint16_t nwqes_mask = __ldg(&txq->wqe_mask);
	const uintptr_t wqe_addr = __ldg((uintptr_t *)&txq->wqe_addr);
	const uint16_t idx = wqe_idx & nwqes_mask;
	return (struct doca_gpu_dev_eth_txq_wqe *)(wqe_addr + (idx << DOCA_GPUNETIO_ETH_MLX5_WQE_SHIFT));
}

/**
 * @brief Reserve a number of WQE slots.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] count - Number of WQE slots to reserve
 * @return The index of the first reserved WQE slot
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>
__device__ static inline uint64_t doca_gpu_dev_eth_txq_reserve_wq_slots(struct doca_gpu_eth_txq *txq, uint32_t count)
{
	uint64_t wqe_idx = doca_gpu_dev_eth_atomic_add<uint64_t, resource_sharing_mode>(&txq->sq_rsvd_index, count);

	return wqe_idx;
}

/**
 * @brief Mark the WQEs in the range [from_wqe_idx, to_wqe_idx] as ready.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] from_wqe_idx - Starting WQE index
 * @param [in] to_wqe_idx - Ending WQE index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>
__device__ static inline void doca_gpu_dev_eth_txq_mark_wqes_ready(struct doca_gpu_eth_txq *txq,
								   uint64_t from_wqe_idx,
								   uint64_t to_wqe_idx)
{
	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE)
		txq->sq_ready_index = to_wqe_idx + 1;
	else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA) {
		doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>();
		cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ready_index_aref(txq->sq_ready_index);
		while (ready_index_aref.load(cuda::std::memory_order_relaxed) != from_wqe_idx)
			continue;
		doca_gpu_dev_eth_fence_acquire<DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA>();
		ready_index_aref.store(to_wqe_idx + 1, cuda::std::memory_order_relaxed);
	} else if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU) {
		doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
		cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ready_index_aref(txq->sq_ready_index);
		while (ready_index_aref.load(cuda::std::memory_order_relaxed) != from_wqe_idx)
			continue;
		doca_gpu_dev_eth_fence_acquire<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
		ready_index_aref.store(to_wqe_idx + 1, cuda::std::memory_order_relaxed);
	}
}

/**
 * @brief Similarly to doca_eth_txq_calculate_timestamp() it returns the wait on time
 * value to use when preparing a wait time WQE.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] timestamp_ns - UTC timestampt to convert
 * @param [out] wait_on_time_ts - Wait on time value
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_NOT_SUPPORTED - wait on time support is not NATIVE
 *
 */
__device__ static inline doca_error_t doca_gpu_dev_eth_txq_calculate_timestamp(const struct doca_gpu_eth_txq *txq,
									       const uint64_t timestamp_ns,
									       uint64_t *wait_on_time_ts)
{
	*wait_on_time_ts = doca_gpu_dev_eth_bswap64((timestamp_ns % (uint64_t)DOCA_GPUNETIO_ETH_NS_PER_S) |
						    ((timestamp_ns / (uint64_t)DOCA_GPUNETIO_ETH_NS_PER_S) << 32));
	return DOCA_SUCCESS;
}

/*********************************************************************************************************************
 * WQE Prepare
 *********************************************************************************************************************/

/**
 * @brief Prepare a send WQE with only 1 data segment at a specific WQE pointer location.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] wqe_ptr - Memory pointer to the WQE to be prepared
 * @param [in] wqe_idx - WQE number
 * @param [in] addr - Send, src address
 * @param [in] mkey - Send, memory key. It must be in network byte order.
 * @param [in] nbytes - Send, number of bytes
 * @param [in] flags - flags from enum doca_gpu_eth_send_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_NOT_SUPPORTED - wait on time support is not NATIVE
 *
 */
__device__ __inline__ static doca_error_t doca_gpu_dev_eth_txq_wqe_prepare_send(
	const struct doca_gpu_eth_txq *txq,
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr,
	const uint16_t wqe_idx,
	const uint64_t addr,
	const uint32_t mkey,
	const uint32_t nbytes,
	const enum doca_gpu_eth_send_flags flags)
{
	struct doca_gpu_dev_eth_wqe_ctrl_seg cseg;
	struct doca_gpu_dev_eth_txq_wqe_eth_seg eseg;
	struct mlx5_wqe_data_seg dseg;

	cseg.opmod_idx_opcode = doca_gpu_dev_eth_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_ETH_WQE_IDX_SHIFT) |
							 DOCA_GPUNETIO_ETH_MLX5_OPCODE_SEND);
	cseg.qpn_ds = __ldg(&(txq->sq_num_shift8_be_1ds));
	if (flags & DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY)
		cseg.flags = txq->const_cflags_always_be;
	else
		cseg.flags = txq->const_cflags_first_err_be;
	cseg.imm = 0;

	eseg.offsets = 0;
	eseg.flags = 0;
	eseg.flow_metadata = 0;
	eseg.inline_hdr = 0;

	if (txq->enable_l3_cs_offload)
		eseg.cs_flags |= DOCA_GPUNETIO_ETH_WQE_L3_CSUM;
	if (txq->enable_l4_cs_offload)
		eseg.cs_flags |= DOCA_GPUNETIO_ETH_WQE_L4_CSUM;

	dseg.byte_count = doca_gpu_dev_eth_bswap32((uint32_t)nbytes);
	dseg.lkey = mkey;
	dseg.addr = doca_gpu_dev_eth_bswap64((uintptr_t)addr);

	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->cseg), (uint64_t *)&(cseg));
	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->eseg), (uint64_t *)&(eseg));
	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(dseg));

	return DOCA_SUCCESS;
}

/**
 * @brief Prepare a send WQE with 2 data segments at a specific WQE pointer location.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] wqe_ptr - Memory pointer to the WQE to be prepared
 * @param [in] wqe_idx - WQE number
 * @param [in] addr0 - Send in first segment, src address
 * @param [in] mkey0 - Send in first segment, memory key. It must be in network byte order.
 * @param [in] nbytes0 - Send in first segment, number of bytes
 * @param [in] addr1 - Send in second segment, src address
 * @param [in] mkey1 - Send in second segment, memory key. It must be in network byte order.
 * @param [in] nbytes1 - Send in second segment, number of bytes
 * @param [in] flags - flags from enum doca_gpu_eth_send_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_NOT_SUPPORTED - wait on time support is not NATIVE
 *
 */
__device__ __inline__ static doca_error_t doca_gpu_dev_eth_txq_wqe_prepare_send(
	const struct doca_gpu_eth_txq *txq,
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr,
	const uint16_t wqe_idx,
	const uint64_t addr0,
	const uint32_t mkey0,
	const uint32_t nbytes0,
	const uint64_t addr1,
	const uint32_t mkey1,
	const uint32_t nbytes1,
	const enum doca_gpu_eth_send_flags flags)
{
	struct mlx5_wqe_data_seg dseg1;

	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, wqe_idx, addr0, mkey0, nbytes0, flags);

	wqe_ptr->cseg.qpn_ds = __ldg(&txq->sq_num_shift8_be_2ds);

	dseg1.byte_count = doca_gpu_dev_eth_bswap32((uint32_t)nbytes1);
	dseg1.lkey = mkey1;
	dseg1.addr = doca_gpu_dev_eth_bswap64((uintptr_t)addr1);

	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg1));

	return DOCA_SUCCESS;
}

/**
 * @brief Prepare a send WQE using the Enhanced Multi-Packet WQE send mode.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] wqe_ptr - Memory pointer to the WQE to be prepared
 * @param [in] wqe_idx - WQE number
 * @param [in] ds_num - Data segment number to fill in the eMPW WQE control segment
 * @param [in] ds_idx - Data segment index to fill in this eMPW WQE
 * @param [in] addr - Send src address
 * @param [in] mkey - Send memory key. It must be in network byte order.
 * @param [in] nbytes - Send number of bytes
 * @param [in] flags - flags from enum doca_gpu_eth_send_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 *
 */
__device__ inline doca_error_t doca_gpu_dev_eth_txq_wqe_prepare_send_empw(const struct doca_gpu_eth_txq *txq,
									  struct doca_gpu_dev_eth_txq_wqe *wqe_ptr,
									  const uint16_t wqe_idx,
									  const int ds_num,
									  const int ds_idx,
									  const uint64_t addr,
									  const uint32_t mkey,
									  const uint32_t nbytes,
									  const enum doca_gpu_eth_send_flags flags)
{
	struct mlx5_wqe_data_seg dseg;

	if (nbytes == 0)
		dseg.byte_count = doca_gpu_dev_eth_bswap32(0x80000000);
	else
		dseg.byte_count = doca_gpu_dev_eth_bswap32((uint32_t)nbytes);

	dseg.lkey = mkey;
	dseg.addr = doca_gpu_dev_eth_bswap64((uintptr_t)addr);

	if (ds_idx == 0) {
		struct doca_gpu_dev_eth_wqe_ctrl_seg cseg;
		struct doca_gpu_dev_eth_txq_wqe_eth_seg eseg = {0};

		cseg.opmod_idx_opcode =
			doca_gpu_dev_eth_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_ETH_WQE_IDX_SHIFT) |
						 DOCA_GPUNETIO_ETH_MLX5_OPCODE_ENHANCED_MPSW);
		cseg.qpn_ds = doca_gpu_dev_eth_bswap32(__ldg(&(txq->sq_num_shift8)) | ((ds_num + 2) & 0x3F));
		if (flags & DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY)
			cseg.flags = txq->const_cflags_always_be;
		else
			cseg.flags = txq->const_cflags_first_err_be;
		cseg.imm = 0;

		doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->cseg), (uint64_t *)&(cseg));
		doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->eseg), (uint64_t *)&(eseg));
		doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(dseg));
	} else if (ds_idx == 1) {
		doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg));
	} else {
		struct doca_gpu_dev_eth_txq_wqe_empw *ds_wqe = (struct doca_gpu_dev_eth_txq_wqe_empw *)wqe_ptr;
		doca_gpu_dev_eth_store_wqe_seg(
			(uint64_t *)&(ds_wqe->dseg[(ds_idx + 2) % DOCA_GPUNETIO_ETH_EMPW_SEGMENTS_PER_WQE]),
			(uint64_t *)&(dseg));
	}

	return DOCA_SUCCESS;
}

/**
 * @brief Prepare a wait on time WQE with only 1 data segment at a specific WQE pointer location.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] wqe_ptr - Memory pointer to the WQE to be prepared
 * @param [in] wqe_idx - Wait on time value
 * @param [in] wait_on_time_ts - Wait on time timestamp to assign
 * @param [in] flags - flags from enum doca_gpu_eth_send_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_NOT_SUPPORTED - wait on time support is not NATIVE
 *
 */
__device__ __inline__ static doca_error_t doca_gpu_dev_eth_txq_wqe_prepare_wait_time(
	const struct doca_gpu_eth_txq *txq,
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr,
	const uint16_t wqe_idx,
	const uint64_t wait_on_time_ts,
	const enum doca_gpu_eth_send_flags flags)
{
	struct doca_gpu_dev_eth_wqe_ctrl_seg cseg;
	struct doca_gpu_dev_eth_txq_wqe_wait_seg wseg;

	cseg.opmod_idx_opcode = doca_gpu_dev_eth_bswap32(
		((uint32_t)wqe_idx << DOCA_GPUNETIO_ETH_WQE_IDX_SHIFT) |
		(DOCA_GPUNETIO_ETH_MLX5_OPCODE_WAIT | (DOCA_GPUNETIO_ETH_MLX5_OPC_MOD_WAIT_TIME << 24)));
	cseg.qpn_ds = __ldg(&txq->const_wait_be);
	if (flags & DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY)
		cseg.flags = txq->const_cflags_always_be;
	else
		cseg.flags = txq->const_cflags_first_err_be;

	wseg.operation = txq->const_wait_wqe_op_be;
	wseg.value = wait_on_time_ts;
	wseg.mask = 0xffffffffffffffff;

	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->cseg), (uint64_t *)&(cseg));
	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->wseg.w0), (uint64_t *)&(wseg.w0));
	doca_gpu_dev_eth_store_wqe_seg((uint64_t *)&(wqe_ptr->wseg.w1), (uint64_t *)&(wseg.w1));

	return DOCA_SUCCESS;
}

/*********************************************************************************************************************
 * Update NIC registers (DB and DBREC)
 *********************************************************************************************************************/

/**
 * @brief [Internal] Update the NIC DBREC (Doorbell Record).
 * This function does not guarantee the ordering.
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 */
__device__ static inline void doca_priv_gpu_dev_eth_txq_update_dbr(struct doca_gpu_eth_txq *txq, uint32_t prod_index)
{
	__be32 dbrec_val = doca_gpu_dev_eth_prepare_dbr(prod_index);
	__be32 *dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&txq->wqe_db_rec);

	cuda::atomic_ref<__be32, cuda::thread_scope_system> dbrec_ptr_aref(*dbrec_ptr);
	dbrec_ptr_aref.store(dbrec_val, cuda::std::memory_order_relaxed);
}

/**
 * @brief Update the NIC DBREC (Doorbell Record)
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_txq_update_dbr(struct doca_gpu_eth_txq *txq, uint32_t prod_index)
{
	doca_gpu_dev_eth_fence_release<sync_scope>();
	doca_priv_gpu_dev_eth_txq_update_dbr(txq, prod_index);
}

/**
 * @brief Prepare the NIC DB (Doorbell)
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 * @return DB value
 */
__device__ static inline __be64 doca_gpu_dev_eth_txq_prepare_db(struct doca_gpu_eth_txq *txq, uint64_t prod_index)
{
	struct doca_gpu_dev_eth_wqe_ctrl_seg ctrl_seg = {0};

	ctrl_seg.qpn_ds = __ldg(&txq->sq_num_shift8_be);
	ctrl_seg.opmod_idx_opcode = doca_gpu_dev_eth_bswap32((prod_index << DOCA_GPUNETIO_ETH_WQE_IDX_SHIFT));

	return *(uint64_t *)&ctrl_seg;
}

/**
 * @brief Submit all the WQEs up to the given producer index to the NIC via the CPU proxy.
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_txq_submit_proxy(struct doca_gpu_eth_txq *txq, uint64_t prod_index)
{
	__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&txq->wqe_db);

	if (resource_sharing_mode == DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE) {
		doca_priv_gpu_dev_eth_txq_update_dbr(txq, prod_index);
		DOCA_GPUNETIO_ETH_WRITE_ONCE(*db_ptr, prod_index);
		DOCA_GPUNETIO_ETH_WRITE_ONCE(txq->wqe_pi, prod_index);
	} else {
		doca_gpu_dev_eth_lock<resource_sharing_mode>(&txq->lock);

		uint64_t old_prod_index =
			doca_gpu_dev_eth_atomic_max<uint64_t, resource_sharing_mode, true>(&txq->wqe_pi, prod_index);

		if (old_prod_index < prod_index) {
			doca_priv_gpu_dev_eth_txq_update_dbr(txq, prod_index);
			cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
			doca_gpu_dev_eth_fence_release<sync_scope>();
			db_ptr_aref.store(prod_index, cuda::std::memory_order_relaxed);
		}

		doca_gpu_dev_eth_unlock<resource_sharing_mode>(&txq->lock);
	}
}

/**
 * @brief Ring the NIC DB (Doorbell)
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_txq_ring_db(struct doca_gpu_eth_txq *txq, uint64_t prod_index)
{
	__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&txq->wqe_db);
	__be64 db_val = doca_gpu_dev_eth_txq_prepare_db(txq, prod_index);

#ifdef DOCA_GPUNETIO_ETH_HAS_STORE_RELAXED_MMIO
	{
		doca_gpu_dev_eth_fence_release<sync_scope>();
		doca_gpu_dev_eth_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_val);
	}
#else
	{
		cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
		doca_gpu_dev_eth_fence_release<sync_scope>();
		db_ptr_aref.store(db_val, cuda::std::memory_order_relaxed);
	}
#endif
}

/**
 * @brief Submit a work request to the NIC using the NIC DB protocol.
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_txq_submit_db(struct doca_gpu_eth_txq *txq, uint64_t prod_index)
{
	doca_gpu_dev_eth_lock<resource_sharing_mode>(&txq->lock);

	uint64_t old_prod_index =
		doca_gpu_dev_eth_atomic_max<uint64_t, resource_sharing_mode, true>(&txq->wqe_pi, prod_index);

	if (old_prod_index < prod_index) {
		// Early rining of the NIC DB to push WQEs to the NIC ASAP.
		doca_gpu_dev_eth_txq_ring_db<sync_scope>(txq, prod_index);

		// In case the recovery path is triggered, the later DB ringing will cover for correctness.
		doca_priv_gpu_dev_eth_txq_update_dbr(txq, prod_index);
		doca_gpu_dev_eth_txq_ring_db<sync_scope>(txq, prod_index);
	}

	doca_gpu_dev_eth_unlock<resource_sharing_mode>(&txq->lock);
}

/**
 * @brief Submit work requests to the NIC using the NIC DB protocol or the proxy.
 *
 * @param txq - GPU Tx Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_submit(struct doca_gpu_eth_txq *txq, uint64_t prod_index)
{
	if (nic_handler == DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO) {
		const enum doca_gpu_dev_eth_nic_handler qp_nic_handler =
			(enum doca_gpu_dev_eth_nic_handler)__ldg((int *)&txq->nic_handler);
		if (qp_nic_handler == DOCA_GPUNETIO_ETH_NIC_HANDLER_GPU_SM_DB)
			doca_gpu_dev_eth_txq_submit_db<resource_sharing_mode, sync_scope>(txq, prod_index);
		else
			doca_gpu_dev_eth_txq_submit_proxy<resource_sharing_mode, sync_scope>(txq, prod_index);
	} else if (nic_handler == DOCA_GPUNETIO_ETH_NIC_HANDLER_GPU_SM_DB) {
		doca_gpu_dev_eth_txq_submit_db<resource_sharing_mode, sync_scope>(txq, prod_index);
	} else {
		doca_gpu_dev_eth_txq_submit_proxy<resource_sharing_mode, sync_scope>(txq, prod_index);
	}
}

/*********************************************************************************************************************
 * Send with shared QP support
 *********************************************************************************************************************/

/**
 * @brief Send shared QP function, thread scope.
 * Every thread posts a different Send in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_send_thread(struct doca_gpu_eth_txq *txq,
							       const uint64_t addr,
							       const uint32_t mkey,
							       const size_t size,
							       enum doca_gpu_eth_send_flags flags,
							       doca_gpu_dev_eth_ticket_t *out_ticket)
{
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr;
	uint64_t wqe_idx;

	DOCA_GPUNETIO_ETH_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_ETH_ASSERT(txq != NULL);

	wqe_idx = doca_gpu_dev_eth_txq_reserve_wq_slots<resource_sharing_mode>(txq, 1);
	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);

	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, wqe_idx, addr, mkey, size, flags);

	doca_gpu_dev_eth_txq_mark_wqes_ready<resource_sharing_mode>(txq, wqe_idx, wqe_idx);
	doca_gpu_dev_eth_txq_submit<resource_sharing_mode, sync_scope, nic_handler>(txq, wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Send shared QP function, warp scope.
 * Every thread in the warp posts a different Send in a different WQE position.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_send_warp(struct doca_gpu_eth_txq *txq,
							     uint64_t addr,
							     uint32_t mkey,
							     size_t size,
							     enum doca_gpu_eth_send_flags flags,
							     doca_gpu_dev_eth_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr;
	uint64_t base_wqe_idx = 0;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t wqe_idx;
	uint32_t lane_idx = doca_gpu_dev_eth_get_lane_id();

	DOCA_GPUNETIO_ETH_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_ETH_ASSERT(txq != NULL);

	if (lane_idx == 0) {
		base_wqe_idx =
			doca_gpu_dev_eth_txq_reserve_wq_slots<resource_sharing_mode>(txq, DOCA_GPUNETIO_ETH_WARP_SIZE);
		base_wqe_idx_0 = (uint32_t)base_wqe_idx;
		base_wqe_idx_1 = (uint32_t)(base_wqe_idx >> 32);
	}
	__syncwarp();

	base_wqe_idx_0 = __reduce_max_sync(DOCA_GPUNETIO_ETH_WARP_FULL_MASK, base_wqe_idx_0);
	base_wqe_idx_1 = __reduce_max_sync(DOCA_GPUNETIO_ETH_WARP_FULL_MASK, base_wqe_idx_1);
	base_wqe_idx = ((uint64_t)base_wqe_idx_1) << 32 | base_wqe_idx_0;

	wqe_idx = base_wqe_idx + lane_idx;
	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);

	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, wqe_idx, addr, mkey, size, flags);
	__syncwarp();
	if (lane_idx == 0) {
		doca_gpu_dev_eth_txq_mark_wqes_ready<resource_sharing_mode>(txq,
									    base_wqe_idx,
									    base_wqe_idx + DOCA_GPUNETIO_ETH_WARP_SIZE -
										    1);
		doca_gpu_dev_eth_txq_submit<resource_sharing_mode, sync_scope, nic_handler>(
			txq,
			base_wqe_idx + DOCA_GPUNETIO_ETH_WARP_SIZE);
	}
	__syncwarp();

	*out_ticket = wqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Send shared QP function, block scope.
 * Every thread in the block posts a different Send in a different WQE position.
 * Only the thread 0 in the block submits all the WQE posted by the threads in the block.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_send_block(struct doca_gpu_eth_txq *txq,
							      uint64_t addr,
							      uint32_t mkey,
							      size_t size,
							      enum doca_gpu_eth_send_flags flags,
							      doca_gpu_dev_eth_ticket_t *out_ticket)
{
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr;
	__shared__ uint64_t base_wqe_idx;

	DOCA_GPUNETIO_ETH_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_ETH_ASSERT(txq != NULL);

	// Assuming 1D CUDA block
	if (threadIdx.x == 0)
		base_wqe_idx = doca_gpu_dev_eth_txq_reserve_wq_slots<resource_sharing_mode>(txq, blockDim.x);
	__syncthreads();

	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, base_wqe_idx + threadIdx.x);

	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, base_wqe_idx + threadIdx.x, addr, mkey, size, flags);
	__syncthreads();
	if (threadIdx.x == 0) {
		doca_gpu_dev_eth_txq_mark_wqes_ready<resource_sharing_mode>(txq,
									    base_wqe_idx,
									    base_wqe_idx + blockDim.x - 1);
		doca_gpu_dev_eth_txq_submit<resource_sharing_mode, sync_scope, nic_handler>(txq,
											    base_wqe_idx + blockDim.x);
	}
	__syncthreads();

	*out_ticket = base_wqe_idx + threadIdx.x;
}

/**
 * @brief Send shared QP switch function. Redirect to thread, warp or block scope.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_eth_exec_scope exec_scope = DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_eth_txq_send(struct doca_gpu_eth_txq *txq,
							uint64_t addr,
							uint32_t mkey,
							size_t size,
							enum doca_gpu_eth_send_flags flags,
							doca_gpu_dev_eth_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD)
		doca_gpu_dev_eth_txq_send_thread<resource_sharing_mode, sync_scope, nic_handler>(txq,
												 addr,
												 mkey,
												 size,
												 flags,
												 out_ticket);
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP)
		doca_gpu_dev_eth_txq_send_warp<resource_sharing_mode, sync_scope, nic_handler>(txq,
											       addr,
											       mkey,
											       size,
											       flags,
											       out_ticket);
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK)
		doca_gpu_dev_eth_txq_send_block<resource_sharing_mode, sync_scope, nic_handler>(txq,
												addr,
												mkey,
												size,
												flags,
												out_ticket);
}

/*********************************************************************************************************************
 * Wait + Send with shared QP support
 *********************************************************************************************************************/

/**
 * @brief Wait + Send shared QP function, thread scope.
 * Every thread posts a sequence of Wait WQE + Send WQE in different positions.
 * Every thread submits the WQE.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] wait_on_time_ts - Timestamp to set in the wait WQE
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_wait_send_thread(struct doca_gpu_eth_txq *txq,
								    const uint64_t wait_on_time_ts,
								    const uint64_t addr,
								    const uint32_t mkey,
								    const size_t size,
								    enum doca_gpu_eth_send_flags flags,
								    doca_gpu_dev_eth_ticket_t *out_ticket)
{
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr;
	uint64_t base_wqe_idx = 0, wqe_idx;

	DOCA_GPUNETIO_ETH_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_ETH_ASSERT(txq != NULL);

	base_wqe_idx = doca_gpu_dev_eth_txq_reserve_wq_slots<resource_sharing_mode>(txq, 2);
	wqe_idx = base_wqe_idx;
	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);
	doca_gpu_dev_eth_txq_wqe_prepare_wait_time(txq,
						   wqe_ptr,
						   wqe_idx,
						   wait_on_time_ts,
						   DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);

	wqe_idx++;

	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);
	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, wqe_idx, addr, mkey, size, flags);

	doca_gpu_dev_eth_txq_mark_wqes_ready<resource_sharing_mode>(txq, base_wqe_idx, wqe_idx);
	doca_gpu_dev_eth_txq_submit<resource_sharing_mode, sync_scope, nic_handler>(txq, wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Wait + Send shared QP function, warp scope.
 * Every thread in the warp posts a sequence of Wait WQE + Send WQE in different positions.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] wait_on_time_ts - Timestamp to set in the wait WQE
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_wait_send_warp(struct doca_gpu_eth_txq *txq,
								  const uint64_t wait_on_time_ts,
								  uint64_t addr,
								  uint32_t mkey,
								  size_t size,
								  enum doca_gpu_eth_send_flags flags,
								  doca_gpu_dev_eth_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr;
	uint64_t base_wqe_idx = 0;
	uint64_t wqe_idx;
	uint32_t lane_idx = doca_gpu_dev_eth_get_lane_id();

	DOCA_GPUNETIO_ETH_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_ETH_ASSERT(txq != NULL);

	if (lane_idx == 0)
		base_wqe_idx =
			doca_gpu_dev_eth_txq_reserve_wq_slots<resource_sharing_mode>(txq,
										     DOCA_GPUNETIO_ETH_WARP_SIZE + 1);
	__syncwarp();

	base_wqe_idx = __reduce_max_sync(DOCA_GPUNETIO_ETH_WARP_FULL_MASK, (uint32_t)base_wqe_idx);

	// First WQE in the warp is the wait
	wqe_idx = base_wqe_idx + lane_idx + 1;
	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);

	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, wqe_idx, addr, mkey, size, flags);
	__syncwarp();
	if (lane_idx == 0) {
		wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, base_wqe_idx);
		doca_gpu_dev_eth_txq_wqe_prepare_wait_time(txq,
							   wqe_ptr,
							   base_wqe_idx,
							   wait_on_time_ts,
							   DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);

		doca_gpu_dev_eth_txq_mark_wqes_ready<resource_sharing_mode>(txq,
									    base_wqe_idx,
									    base_wqe_idx + DOCA_GPUNETIO_ETH_WARP_SIZE);
		doca_gpu_dev_eth_txq_submit<resource_sharing_mode, sync_scope, nic_handler>(
			txq,
			base_wqe_idx + DOCA_GPUNETIO_ETH_WARP_SIZE + 1);
	}
	__syncwarp();

	*out_ticket = wqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Wait + Send shared QP function, block scope.
 * Every thread in the block posts a sequence of Wait WQE + Send WQE in different positions.
 * Only the thread 0 in the block submits all the WQE posted by the threads in the block.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] wait_on_time_ts - Timestamp to set in the wait WQE
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_eth_txq_wait_send_block(struct doca_gpu_eth_txq *txq,
								   const uint64_t wait_on_time_ts,
								   uint64_t addr,
								   uint32_t mkey,
								   size_t size,
								   enum doca_gpu_eth_send_flags flags,
								   doca_gpu_dev_eth_ticket_t *out_ticket)
{
	struct doca_gpu_dev_eth_txq_wqe *wqe_ptr;
	__shared__ uint64_t base_wqe_idx;
	uint64_t wqe_idx = 0;

	DOCA_GPUNETIO_ETH_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_ETH_ASSERT(txq != NULL);

	// Assuming 1D CUDA block
	if (threadIdx.x == 0)
		base_wqe_idx = doca_gpu_dev_eth_txq_reserve_wq_slots<resource_sharing_mode>(txq, blockDim.x + 1);
	__syncthreads();

	// First WQE in the block is the wait
	wqe_idx = base_wqe_idx + threadIdx.x + 1;
	wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, wqe_idx);

	doca_gpu_dev_eth_txq_wqe_prepare_send(txq, wqe_ptr, wqe_idx, addr, mkey, size, flags);
	__syncthreads();
	if (threadIdx.x == 0) {
		wqe_ptr = doca_gpu_dev_eth_txq_get_wqe_ptr(txq, base_wqe_idx);
		doca_gpu_dev_eth_txq_wqe_prepare_wait_time(txq,
							   wqe_ptr,
							   base_wqe_idx,
							   wait_on_time_ts,
							   DOCA_GPUNETIO_ETH_SEND_FLAG_NONE);

		doca_gpu_dev_eth_txq_mark_wqes_ready<resource_sharing_mode>(txq,
									    base_wqe_idx,
									    base_wqe_idx + blockDim.x);
		doca_gpu_dev_eth_txq_submit<resource_sharing_mode, sync_scope, nic_handler>(txq,
											    base_wqe_idx + blockDim.x +
												    1);
	}
	__syncthreads();

	*out_ticket = wqe_idx;
}

/**
 * @brief Wait + Send shared QP switch function. Redirect to thread, warp or block scope.
 *
 * @param [in] txq - GPU Txq Queue Pair (QP)
 * @param [in] wait_on_time_ts - Timestamp to set in the wait WQE
 * @param [in] addr - src address
 * @param [in] mkey - memory key. It must be in network byte order.
 * @param [in] size - number of bytes
 * @param [in] flags - Flags from enum doca_gpu_eth_send_flags
 * @param [in] out_ticket - WQE index of the Send
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_eth_exec_scope exec_scope = DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_eth_txq_wait_send(struct doca_gpu_eth_txq *txq,
							     const uint64_t wait_on_time_ts,
							     uint64_t addr,
							     uint32_t mkey,
							     size_t size,
							     enum doca_gpu_eth_send_flags flags,
							     doca_gpu_dev_eth_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD)
		doca_gpu_dev_eth_txq_wait_send_thread<resource_sharing_mode, sync_scope, nic_handler>(txq,
												      wait_on_time_ts,
												      addr,
												      mkey,
												      size,
												      flags,
												      out_ticket);
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP)
		doca_gpu_dev_eth_txq_wait_send_warp<resource_sharing_mode, sync_scope, nic_handler>(txq,
												    wait_on_time_ts,
												    addr,
												    mkey,
												    size,
												    flags,
												    out_ticket);
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK)
		doca_gpu_dev_eth_txq_wait_send_block<resource_sharing_mode, sync_scope, nic_handler>(txq,
												     wait_on_time_ts,
												     addr,
												     mkey,
												     size,
												     flags,
												     out_ticket);
}

/*********************************************************************************************************************
 * Completion Queue (CQ)
 *********************************************************************************************************************/

/**
 * @brief Submit the NIC DBREC (Doorbell Record) for CQ
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_txq_submit_cq_dbr(struct doca_gpu_eth_txq *txq, uint64_t prod_index)
{
	__be32 dbrec_val;
	__be32 *dbrec_ptr;

	doca_gpu_dev_eth_lock<resource_sharing_mode>(&txq->cq_lock);

	uint64_t old_prod_index =
		doca_gpu_dev_eth_atomic_max<uint64_t, resource_sharing_mode, true>(&txq->cqe_ci, prod_index);
	if (old_prod_index < prod_index) {
		doca_gpu_dev_eth_fence_release<sync_scope>();

		dbrec_val = doca_gpu_dev_eth_prepare_cq_dbr(prod_index);
		dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&txq->cq_db_rec);

		cuda::atomic_ref<__be32, cuda::thread_scope_system> dbrec_ptr_aref(*dbrec_ptr);
		dbrec_ptr_aref.store(dbrec_val, cuda::std::memory_order_relaxed);
	}

	doca_gpu_dev_eth_unlock<resource_sharing_mode>(&txq->cq_lock);
}

/**
 * @brief Reserve a number of CQE slots the thread will poll.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] count - Number of CQE slots to reserve
 * @return The index of the first reserved CQE slot
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU>
__device__ static inline uint64_t doca_gpu_dev_eth_txq_reserve_cq_slots(struct doca_gpu_eth_txq *txq, uint32_t count)
{
	uint64_t wqe_idx = doca_gpu_dev_eth_atomic_add<uint64_t, resource_sharing_mode>(&txq->cq_rsvd_index, count);

	return wqe_idx;
}

/**
 * @brief Poll on a number of CQE slots waiting for an ok CQE or an erroneous CQE.
 * This function has only thread scope, so each thread can poll on a different subset of CQEs
 * reserved via the doca_gpu_dev_eth_txq_reserve_cq_slots().
 *
 * Ok CQEs are returned by the NIC only if the DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY
 * has been set when posting the send or wait WQE.
 *
 * Depending on the cqe_poll template parameter, this function can poll all the num_cqe slots
 * or just the last one. If the last CQE appear, it means all previous CQEs have been arrived.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] num_cqe - Number of CQE slots to poll
 * @param [in] wait_mode - Poll all CQEs or just the last one
 * @param [out] num_completed - Number of CQEs detected
 */
template <enum doca_gpu_dev_eth_cq_poll_mode cqe_poll = DOCA_GPUNETIO_ETH_CQ_POLL_ALL,
	  enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline doca_error_t doca_gpu_dev_eth_txq_poll_completion(struct doca_gpu_eth_txq *txq,
									   const uint32_t num_cqe,
									   enum doca_gpu_dev_eth_wait_flags wait_mode,
									   uint32_t *num_completed)
{
	struct mlx5_cqe64 *cqe = (struct mlx5_cqe64 *)__ldg((uintptr_t *)&txq->cqe_addr);
	const uint32_t cqe_num = __ldg(&txq->cqe_num);
	const uint32_t cqe_mask = __ldg(&txq->cqe_mask);
	uint32_t cqe_cnt = 0;
	uint8_t opown, opcode;
	volatile struct mlx5_cqe64 *cqe64;
	DOCA_GPUNETIO_ETH_ASSERT(num_cqe > 0);

	uint64_t cqe_ci = doca_gpu_dev_eth_txq_reserve_cq_slots(txq, num_cqe);

	if (cqe_poll == DOCA_GPUNETIO_ETH_CQ_POLL_LAST) {
		cqe_cnt = num_cqe - 1;
		cqe_ci += cqe_cnt;
	}

	cqe64 = (volatile struct mlx5_cqe64 *)&cqe[cqe_ci & cqe_mask];

	do {
		opown = doca_gpu_dev_eth_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
		opcode = opown >> DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT;
		if ((opcode != MLX5_CQE_INVALID) && !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cqe_ci & cqe_num))) {
			if (opcode == (uint8_t)MLX5_CQE_REQ_ERR) {
				doca_gpu_dev_eth_print_cqe_err(cqe64);
				return DOCA_ERROR_UNEXPECTED;
			}

			cqe_ci++;
			cqe_cnt++;
			cqe64 = (volatile struct mlx5_cqe64 *)&cqe[cqe_ci & cqe_mask];
		} else if (wait_mode == DOCA_GPUNETIO_ETH_WAIT_FLAG_NB)
			break;
	} while (cqe_cnt < num_cqe);

	if (cqe_cnt > 0)
		doca_gpu_dev_eth_txq_submit_cq_dbr<resource_sharing_mode, sync_scope>(txq, cqe_ci);

	*(num_completed) = cqe_cnt;

	return DOCA_SUCCESS;
}

/**
 * @brief Poll on a specific CQE slots waiting for an ok CQE or an erroneous CQE.
 * This function doesn't have any lock or reserve capability so the application is in charge
 * to avoid that two threads are polling on the same CQE slot at the same time.
 *
 * @param [in] txq - GPU Tx Queue Pair (QP)
 * @param [in] num_cqe - Number of CQE slots to poll
 * @param [in] wait_mode - Poll all CQEs or just the last one
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline doca_error_t doca_gpu_dev_eth_txq_poll_completion_at(struct doca_gpu_eth_txq *txq,
									      const uint64_t cqe_idx,
									      enum doca_gpu_dev_eth_wait_flags wait_mode)
{
	struct mlx5_cqe64 *cqe = (struct mlx5_cqe64 *)__ldg((uintptr_t *)&txq->cqe_addr);
	const uint32_t cqe_num = __ldg(&txq->cqe_num);
	const uint32_t cqe_mask = __ldg(&txq->cqe_mask);
	uint64_t cqe_ci = cqe_idx;
	uint8_t opown, opcode;
	volatile struct mlx5_cqe64 *cqe64 = (volatile struct mlx5_cqe64 *)&cqe[cqe_ci & cqe_mask];

	do {
		opown = doca_gpu_dev_eth_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
		opcode = opown >> DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT;
		if ((opcode != MLX5_CQE_INVALID) && !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cqe_ci & cqe_num))) {
			if (opcode == (uint8_t)MLX5_CQE_REQ_ERR) {
				doca_gpu_dev_eth_print_cqe_err(cqe64);
				return DOCA_ERROR_UNEXPECTED;
			}

			cqe_ci++;
		} else if (wait_mode == DOCA_GPUNETIO_ETH_WAIT_FLAG_NB)
			break;
	} while (cqe_ci == cqe_idx);

	if (cqe_ci != cqe_idx)
		doca_gpu_dev_eth_txq_submit_cq_dbr<resource_sharing_mode, sync_scope>(txq, cqe_ci);

	return DOCA_SUCCESS;
}

#endif /* DOCA_GPUNETIO_DEVICE_ETH_TXQ_H */

/** @} */
