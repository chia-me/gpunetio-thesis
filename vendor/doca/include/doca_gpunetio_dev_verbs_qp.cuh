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
 * @file doca_gpunetio_dev_verbs_qp.cuh
 * @page DOCA GPUNetIO CUDA Device functions with src exposed
 * @defgroup DOCA_GPUNETIO_DEV_DEF DOCA GPUNetIO Device - Definitions
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */

#ifndef DOCA_GPUNETIO_DEV_VERBS_QP_H
#define DOCA_GPUNETIO_DEV_VERBS_QP_H

#include <cuda/atomic>
#include <doca_gpunetio_dev_verbs_cq.cuh>

/* *********** WQE UTILS *********** */
__device__ static inline void doca_gpu_dev_verbs_store_wqe_seg(uint64_t *ptr, uint64_t *val)
{
	asm volatile("st.weak.cs.v2.b64 [%0], {%1, %2};" : : "l"(ptr), "l"(val[0]), "l"(val[1]));
}

/**
 * @brief Get a pointer to the WQE buffer at a specific index
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_idx - Index of the WQE to get
 * @return Pointer to the WQE buffer at the specified index
 */
__device__ static inline struct doca_gpu_dev_verbs_wqe *doca_gpu_dev_verbs_get_wqe_ptr(struct doca_gpu_dev_verbs_qp *qp,
										       uint16_t wqe_idx)
{
	const uint16_t nwqes_mask = __ldg(&qp->sq_wqe_mask);
	const uintptr_t wqe_addr = __ldg((uintptr_t *)&qp->sq_wqe_daddr);
	const uint16_t idx = wqe_idx & nwqes_mask;
	return (struct doca_gpu_dev_verbs_wqe *)(wqe_addr + (idx << DOCA_GPUNETIO_MLX5_WQE_SQ_SHIFT));
}

/**
 * @brief Get a pointer to the WQE buffer at a specific index
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_idx - Index of the WQE to get
 * @return Pointer to the WQE buffer at the specified index
 */
__device__ static inline struct mlx5_wqe_data_seg *doca_gpu_dev_verbs_get_rwqe_ptr(struct doca_gpu_dev_verbs_qp *qp,
										   uint16_t wqe_idx)
{
	const uint16_t nwqes_mask = __ldg(&qp->rq_wqe_mask);
	const uintptr_t wqe_addr = __ldg((uintptr_t *)&qp->rq_wqe_daddr);
	const uint16_t idx = wqe_idx & nwqes_mask;
	return (struct mlx5_wqe_data_seg *)(wqe_addr + (idx << DOCA_GPUNETIO_MLX5_WQE_RQ_SHIFT));
}

/**
 * @brief Increment and round up WQE id
 *
 * @param[in] wqe_idx - wqe idx
 * @param[in] increment - wqe idx increment
 *
 * @return wqe incremented idx
 */
__device__ inline uint32_t doca_gpu_dev_verbs_wqe_idx_inc_mask(uint32_t wqe_idx, uint32_t increment)
{
	return (wqe_idx + increment) & DOCA_GPUNETIO_VERBS_WQE_PI_MASK;
}

/**
 * @brief Wait until the given WQE slot is available.
 * All prior WQE slots are also guaranteed to be available.
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_idx - WQE slot index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_wait_until_slot_available(struct doca_gpu_dev_verbs_qp *qp,
									   uint64_t wqe_idx)
{
	uint16_t nwqes;
	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
		nwqes = __ldg(&qp->rq_wqe_num);
		if (wqe_idx >= nwqes)
			doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(
				doca_gpu_dev_verbs_qp_get_cq_rq(qp),
				wqe_idx - nwqes);
	} else {
		nwqes = __ldg(&qp->sq_wqe_num);
		if (wqe_idx >= nwqes)
			doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(
				doca_gpu_dev_verbs_qp_get_cq_sq(qp),
				wqe_idx - nwqes);
	}
}

/**
 * @brief Reserve a number of WQE slots.
 *
 * @param qp - Queue Pair (QP)
 * @param count - Number of WQE slots to reserve
 * @return The index of the first reserved WQE slot
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ,
	  bool wait_for_availability = true>
__device__ static inline uint64_t doca_gpu_dev_verbs_reserve_wq_slots(struct doca_gpu_dev_verbs_qp *qp, uint32_t count)
{
	uint64_t wqe_idx;
	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ)
		wqe_idx = doca_gpu_dev_verbs_atomic_add<uint64_t, resource_sharing_mode>(&qp->rq_rsvd_index, count);
	else
		wqe_idx = doca_gpu_dev_verbs_atomic_add<uint64_t, resource_sharing_mode>(&qp->sq_rsvd_index, count);

	if (wait_for_availability)
		doca_gpu_dev_verbs_wait_until_slot_available<resource_sharing_mode, qp_type>(qp, wqe_idx + count - 1);

	return wqe_idx;
}

/**
 * @brief Mark the WQEs in the range [from_wqe_idx, to_wqe_idx] as ready.
 *
 * @param qp - Queue Pair (QP)
 * @param from_wqe_idx - Starting WQE index
 * @param to_wqe_idx - Ending WQE index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_mark_wqes_ready(struct doca_gpu_dev_verbs_qp *qp,
								 uint64_t from_wqe_idx,
								 uint64_t to_wqe_idx)
{
	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
		if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
			qp->rq_ready_index = to_wqe_idx + 1;
		else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
			doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
			cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ready_index_aref(qp->rq_ready_index);
			while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx)
				continue;
			doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
			ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
		} else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
			doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
			cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ready_index_aref(qp->rq_ready_index);
			while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx)
				continue;
			doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
			ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
		}
	} else {
		if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
			qp->sq_ready_index = to_wqe_idx + 1;
		else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
			doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
			cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ready_index_aref(qp->sq_ready_index);
			while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx)
				continue;
			doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
			ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
		} else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
			doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
			cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ready_index_aref(qp->sq_ready_index);
			while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx)
				continue;
			doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
			ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
		}
	}
}

/* *********** QP DBR/DB *********** */

/**
 * @brief Prepare the DBR (Doorbell Record)
 *
 * @param prod_index - Producer index
 * @return DBR value
 */
__device__ static inline __be32 doca_gpu_dev_verbs_prepare_dbr(uint32_t prod_index)
{
	__be32 dbrec_val;

	// This is equivalent to
	// HTOBE32(dbrec_head & 0xffff);
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
 * @brief [Internal] Update the NIC DBR (Doorbell Record).
 * This function does not guarantee the ordering.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_priv_gpu_dev_verbs_update_dbr(struct doca_gpu_dev_verbs_qp *qp, uint32_t prod_index)
{
	__be32 dbrec_val = doca_gpu_dev_verbs_prepare_dbr(prod_index);
	__be32 *dbrec_ptr;

	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ)
		dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&qp->rq_dbrec);
	else
		dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&qp->sq_dbrec);

	cuda::atomic_ref<__be32, cuda::thread_scope_system> dbrec_ptr_aref(*dbrec_ptr);
	dbrec_ptr_aref.store(dbrec_val, cuda::memory_order_relaxed);
}

/**
 * @brief Update the NIC DBR (Doorbell Record)
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_update_dbr(struct doca_gpu_dev_verbs_qp *qp, uint32_t prod_index)
{
	__be32 dbrec_val = doca_gpu_dev_verbs_prepare_dbr(prod_index);
	__be32 *dbrec_ptr;

	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ)
		dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&qp->rq_dbrec);
	else
		dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&qp->sq_dbrec);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
	if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
		doca_gpu_dev_verbs_async_store_release<sync_scope>(dbrec_ptr, dbrec_val);
	} else
#endif
	{
		doca_gpu_dev_verbs_fence_release<sync_scope>();
		doca_priv_gpu_dev_verbs_update_dbr<qp_type>(qp, prod_index);
	}
}

/**
 * @brief Prepare the DB (Doorbell)
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 * @return DB value
 */
__device__ static inline __be64 doca_gpu_dev_verbs_prepare_db(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg ctrl_seg = {0};

	// The only ctrl segment fields that are inspected while ringing
	// the DB are QP number and WQE index
	ctrl_seg.qpn_ds = __ldg(&qp->sq_num_shift8_be);
	ctrl_seg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32((prod_index << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT));

	return *(uint64_t *)&ctrl_seg;
}

/* *************************** Ring Doorbell *************************** */

/**
 * @brief Ring the DB (Doorbell)
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>
__device__ static inline void doca_gpu_dev_verbs_ring_db(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index)
{
	__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qp->sq_db);
	__be64 db_val = doca_gpu_dev_verbs_prepare_db(qp, prod_index);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
	if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
		doca_gpu_dev_verbs_async_store_release<sync_scope>((uint64_t *)db_ptr, (uint64_t)db_val);
	} else
#endif
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
	{
		doca_gpu_dev_verbs_fence_release<sync_scope>();
		doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_val);
	}
#else
	{
		cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
		doca_gpu_dev_verbs_fence_release<sync_scope>();
		db_ptr_aref.store(db_val, cuda::memory_order_relaxed);
	}
#endif
}

#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
/**
 * @brief Ring the BF (BlueFlame). Requires shared memory.
 *
 * @param qp - Queue Pair (QP)
 * @param wqe - WQE to be ringed. This buffer must be in shared memory.
 */
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_verbs_ring_bf(struct doca_gpu_dev_verbs_qp *qp,
							 struct doca_gpu_dev_verbs_wqe *wqe_ptr)
{
	void *bf_ptr = (void *)__ldg((uintptr_t *)&qp->sq_db);
	uint64_t *wqe = (uint64_t *)wqe_ptr;

	doca_gpu_dev_verbs_fence_release<sync_scope>();
	asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 64;"
				 :
				 : "l"(bf_ptr), "l"(wqe));
}
#endif

/**
 * @brief Ring the BF (BlueFlame). Requires at least 8 threads in the warp.
 *
 * @param qp - Queue Pair (QP)
 * @param wqe - WQE to be ringed
 */
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_verbs_ring_bf_warp(struct doca_gpu_dev_verbs_qp *qp,
							      struct doca_gpu_dev_verbs_wqe *wqe_ptr)
{
	unsigned int lane_id = doca_gpu_dev_verbs_get_lane_id();
	uint64_t *bf_ptr = (uint64_t *)qp->sq_db;
	uint64_t *wqe = (uint64_t *)wqe_ptr;

	if (lane_id == 0)
		doca_gpu_dev_verbs_fence_release<sync_scope>();
	__syncwarp();

	if (lane_id < 8) {
		bf_ptr[lane_id] = wqe[lane_id];
	}
}

/**
 * @brief Ring the proxy.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_idx - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static inline void doca_gpu_dev_verbs_ring_proxy(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_idx)
{
	__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qp->sq_db);

	if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
		DOCA_GPUNETIO_VERBS_WRITE_ONCE(*db_ptr, prod_idx);
		DOCA_GPUNETIO_VERBS_WRITE_ONCE(qp->sq_wqe_pi, prod_idx);
	} else {
		doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);

		uint64_t old_prod_idx =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->sq_wqe_pi, prod_idx);

		if (old_prod_idx < prod_idx) {
			cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
			doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
			db_ptr_aref.store(prod_idx, cuda::std::memory_order_relaxed);
		}

		doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
	}
}

/**
 * @brief Submit a work request to the NIC using the DB protocol.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_submit_db(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index)
{
	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
		doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->rq_lock);

		uint64_t old_prod_index =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->rq_wqe_pi,
											     prod_index);
		if (old_prod_index < prod_index)
			doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_RQ>(qp, prod_index);

		doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->rq_lock);
	} else {
		doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);

		uint64_t old_prod_index =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->sq_wqe_pi,
											     prod_index);
		if (old_prod_index < prod_index) {
			// Early rining of the DB to push WQEs to the NIC ASAP.
			doca_gpu_dev_verbs_ring_db<sync_scope, code_opt>(qp, prod_index);

			// In case the recovery path is triggered, the later DB ringing will cover for
			// correctness.
			doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, prod_index);
			doca_gpu_dev_verbs_ring_db<sync_scope, code_opt>(qp, prod_index);
		}

		doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
	}
}

/**
 * @brief Submit a work request to the NIC using the DB protocol without updating DBREC.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_submit_db_no_dbr(struct doca_gpu_dev_verbs_qp *qp,
								   uint64_t prod_index)
{
	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
		doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->rq_lock);

		uint64_t old_prod_index =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->rq_wqe_pi,
											     prod_index);
		if (old_prod_index < prod_index)
			doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_RQ>(qp, prod_index);

		doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->rq_lock);
	} else {
		uint64_t old_prod_index =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->sq_wqe_pi,
											     prod_index);
		if (old_prod_index < prod_index)
			doca_gpu_dev_verbs_ring_db<sync_scope, code_opt>(qp, prod_index);
	}
}

/**
 * @brief Submit a work request to the NIC using the BlueFlame protocol.
 * This function requires a single thread. Users must pass a pointer to a WQE stored in shared
 * memory. Hopper or a newer generation is required to leaverage the BlueFlame protocol.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 * @param smem_wqe - WQE to be submitted directly to the NIC. The buffer must be in shared memory.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>
__device__ static inline void doca_gpu_dev_verbs_submit_bf(struct doca_gpu_dev_verbs_qp *qp,
							   uint64_t prod_index,
							   struct doca_gpu_dev_verbs_wqe *smem_wqe)
{
#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
	doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);
	unsigned long long int old_prod_index =
		doca_gpu_dev_verbs_atomic_max<unsigned long long int, resource_sharing_mode, true>(
			(unsigned long long int *)&qp->sq_wqe_pi,
			(unsigned long long int)prod_index);
	if (old_prod_index < prod_index) {
		doca_gpu_dev_verbs_ring_bf<sync_scope>(qp, smem_wqe);
		doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, prod_index);
		doca_gpu_dev_verbs_ring_db<sync_scope, code_opt>(qp, prod_index);
	}
	doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
#else
	doca_gpu_dev_verbs_submit_db<resource_sharing_mode, sync_scope, code_opt, DOCA_GPUNETIO_VERBS_QP_SQ>(
		qp,
		prod_index);
#endif
}

/**
 * @brief Submit all the WQEs up to the given producer index to the NIC using the BlueFlame
 * protocol. This function must be called by all threads in the warp. At least 8 threads are
 * required.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 * @param wqe - WQE to be submitted directly to the NIC
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>
__device__ static inline void doca_gpu_dev_verbs_submit_bf_warp(struct doca_gpu_dev_verbs_qp *qp,
								uint64_t prod_index,
								struct doca_gpu_dev_verbs_wqe *wqe)
{
	unsigned int lane_id = doca_gpu_dev_verbs_get_lane_id();
	unsigned long long int old_prod_index;
	if (lane_id == 0) {
		doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);
		old_prod_index = doca_gpu_dev_verbs_atomic_max<unsigned long long int, resource_sharing_mode, true>(
			(unsigned long long int *)&qp->sq_wqe_pi,
			(unsigned long long int)prod_index);
	}
	__syncwarp();
	old_prod_index = __shfl_sync(0xFFFFFFFF, old_prod_index, 0);
	if (old_prod_index < prod_index) {
		doca_gpu_dev_verbs_ring_bf_warp(qp, wqe);
		__syncwarp();
		if (lane_id == 0) {
			doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, prod_index);
			doca_gpu_dev_verbs_ring_db<sync_scope, code_opt>(qp, prod_index);
		}
	}
	if (lane_id == 0)
		doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
	__syncwarp();
}

/**
 * @brief Submit all the WQEs up to the given producer index to the NIC via the CPU proxy.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_verbs_submit_proxy(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index)
{
	doca_gpu_dev_verbs_fence_release<sync_scope>();
	doca_gpu_dev_verbs_ring_proxy<resource_sharing_mode>(qp, prod_index);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_submit(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index)
{
	if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
		const enum doca_gpu_dev_verbs_nic_handler qp_nic_handler =
			(enum doca_gpu_dev_verbs_nic_handler)__ldg((int *)&qp->nic_handler);
		if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
			doca_gpu_dev_verbs_submit_db<resource_sharing_mode,
						     sync_scope,
						     DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
						     qp_type>(qp, prod_index);
		else if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
			doca_gpu_dev_verbs_submit_db_no_dbr<resource_sharing_mode,
							     sync_scope,
							     DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
							     qp_type>(qp, prod_index);
		else
			doca_gpu_dev_verbs_submit_proxy<resource_sharing_mode, sync_scope>(qp, prod_index);
	} else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
		doca_gpu_dev_verbs_submit_db<resource_sharing_mode,
					     sync_scope,
					     DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
					     qp_type>(qp, prod_index);
	} else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR) {
		doca_gpu_dev_verbs_submit_db_no_dbr<resource_sharing_mode,
						     sync_scope,
						     DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
						     qp_type>(qp, prod_index);
	} else {
		doca_gpu_dev_verbs_submit_proxy<resource_sharing_mode, sync_scope>(qp, prod_index);
	}
}

/* *********** WQE Types *********** */
/**
 * @brief Prepare an Write WQE with 1 data segment
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_write(struct doca_gpu_dev_verbs_qp *qp,
							    struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							    const uint16_t wqe_idx,
							    const uint32_t opcode,
							    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							    const uint32_t immediate,
							    const uint64_t raddr,
							    const uint32_t rkey,
							    const uint64_t laddr0,
							    const uint32_t lkey0,
							    const uint32_t bytes0)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_raddr_seg rseg;
	struct mlx5_wqe_data_seg dseg0;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_3ds);
	cseg.fm_ce_se = ctrl_flags;
	cseg.imm = immediate;

	rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rseg.rkey = rkey;
#else
	rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
}

/**
 * @brief Prepare an Write WQE with 2 data segments
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_write(struct doca_gpu_dev_verbs_qp *qp,
							    struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							    const uint16_t wqe_idx,
							    const uint32_t opcode,
							    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							    const uint32_t immediate,
							    const uint64_t raddr,
							    const uint32_t rkey,
							    const uint64_t laddr0,
							    const uint32_t lkey0,
							    const uint32_t bytes0,
							    const uint64_t laddr1,
							    const uint32_t lkey1,
							    const uint32_t bytes1)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_raddr_seg rseg;
	struct mlx5_wqe_data_seg dseg0;
	struct mlx5_wqe_data_seg dseg1;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
	cseg.fm_ce_se = ctrl_flags;
	cseg.imm = immediate;

	rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rseg.rkey = rkey;
#else
	rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	dseg1.byte_count = doca_gpu_dev_verbs_bswap32(bytes1);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg1.lkey = lkey1;
#else
	dseg1.lkey = doca_gpu_dev_verbs_bswap32(lkey1);
#endif
	dseg1.addr = doca_gpu_dev_verbs_bswap64(laddr1);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg1));
}

/**
 * @brief Prepare the header segment of an inline RDMA Write WQE.
 * The data segment is prepared separately.
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ static inline void doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(
	struct doca_gpu_dev_verbs_qp *qp,
	struct doca_gpu_dev_verbs_wqe *wqe_ptr,
	const uint16_t wqe_idx,
	enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
	const uint64_t raddr,
	const uint32_t rkey,
	const uint32_t bytes)
{
	int ds;
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_raddr_seg rseg;

	if (bytes > sizeof(struct mlx5_wqe_data_seg) - sizeof(struct mlx5_wqe_inl_data_seg))
		ds = DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MAX;
	else
		ds = DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MIN;

	assert(bytes <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE);

	cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
							   DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE);
	cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | ds);
	cseg.fm_ce_se = ctrl_flags;
	// cseg.imm = 0;

	rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rseg.rkey = rkey;
#else
	rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
}

/**
 * @brief Prepare the data segment of an inline RDMA Write WQE.
 *
 * @param qp - Queue Pair (QP)
 * @param data - Data to be written
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
template <typename T>
__device__ static inline void doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data(struct doca_gpu_dev_verbs_qp *qp,
										 struct doca_gpu_dev_verbs_wqe *wqe_ptr,
										 T data)
{
	struct mlx5_wqe_inl_data_seg *data_seg_ptr =
		(struct mlx5_wqe_inl_data_seg *)((uintptr_t)wqe_ptr + sizeof(struct doca_gpu_dev_verbs_wqe_ctrl_seg) +
						 sizeof(struct mlx5_wqe_raddr_seg));
	struct mlx5_wqe_inl_data_seg data_seg;
	uint32_t bytes = sizeof(T);

	data_seg.byte_count = doca_gpu_dev_verbs_bswap32(bytes | MLX5_INLINE_SEG);
	*(uint32_t *)data_seg_ptr = data_seg.byte_count;

	if (bytes <= sizeof(uint32_t)) {
		T *dst = (T *)((uintptr_t)data_seg_ptr + sizeof(data_seg));
		*dst = data;
	} else {
		uint32_t *dst32 = (uint32_t *)((uintptr_t)data_seg_ptr + sizeof(data_seg));
		dst32[0] = ((uint32_t *)&data)[0];
		dst32[1] = ((uint32_t *)&data)[1];
	}
}

/**
 * @brief Prepare a RDMA Write WQE with inline data
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ static inline void doca_gpu_dev_verbs_wqe_prepare_write_inl(struct doca_gpu_dev_verbs_qp *qp,
								       struct doca_gpu_dev_verbs_wqe *wqe_ptr,
								       const uint16_t wqe_idx,
								       enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
								       const uint64_t raddr,
								       const uint32_t rkey,
								       const uint64_t laddr,
								       const uint32_t bytes)
{
	struct mlx5_wqe_inl_data_seg data_seg;
	struct mlx5_wqe_inl_data_seg *data_seg_ptr =
		(struct mlx5_wqe_inl_data_seg *)((uintptr_t)wqe_ptr + sizeof(struct doca_gpu_dev_verbs_wqe_ctrl_seg) +
						 sizeof(struct mlx5_wqe_raddr_seg));

	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp, wqe_ptr, wqe_idx, ctrl_flags, raddr, rkey, bytes);

	data_seg.byte_count = doca_gpu_dev_verbs_bswap32(bytes | MLX5_INLINE_SEG);
	*(uint32_t *)data_seg_ptr = data_seg.byte_count;

	doca_gpu_dev_verbs_memcpy_data((void *)((uintptr_t)data_seg_ptr + sizeof(data_seg)),
				       (void *)(uintptr_t)laddr,
				       bytes);
}

__device__ inline void doca_gpu_dev_verbs_wqe_prepare_read(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							   const uint16_t wqe_idx,
							   enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							   const uint64_t raddr,
							   const uint32_t rkey,
							   const uint64_t laddr0,
							   const uint32_t lkey0,
							   const uint32_t bytes0)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_raddr_seg rseg;
	struct mlx5_wqe_data_seg dseg0;

	cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
							   DOCA_GPUNETIO_MLX5_OPCODE_RDMA_READ);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_3ds);
	cseg.fm_ce_se = ctrl_flags;

	rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rseg.rkey = rkey;
#else
	rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
}

__device__ inline void doca_gpu_dev_verbs_wqe_prepare_read(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							   const uint16_t wqe_idx,
							   enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							   const uint64_t raddr,
							   const uint32_t rkey,
							   const uint64_t laddr0,
							   const uint32_t lkey0,
							   const uint32_t bytes0,
							   const uint64_t laddr1,
							   const uint32_t lkey1,
							   const uint32_t bytes1)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_raddr_seg rseg;
	struct mlx5_wqe_data_seg dseg0;
	struct mlx5_wqe_data_seg dseg1;

	cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
							   DOCA_GPUNETIO_MLX5_OPCODE_RDMA_READ);
	cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 4);
	cseg.fm_ce_se = ctrl_flags;
	// cseg.imm = 0;

	rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rseg.rkey = rkey;
#else
	rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	dseg1.byte_count = doca_gpu_dev_verbs_bswap32(bytes1);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg1.lkey = lkey1;
#else
	dseg1.lkey = doca_gpu_dev_verbs_bswap32(lkey1);
#endif
	dseg1.addr = doca_gpu_dev_verbs_bswap64(laddr1);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg1));
}

/**
 * @brief Prepare an Atomic WQE
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ static inline void doca_gpu_dev_verbs_wqe_prepare_atomic(struct doca_gpu_dev_verbs_qp *qp,
								    struct doca_gpu_dev_verbs_wqe *wqe_ptr,
								    const uint16_t wqe_idx,
								    const uint32_t opcode,
								    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
								    const uint64_t raddr,
								    const uint32_t rkey,
								    const uint64_t laddr,
								    const uint32_t lkey,
								    const uint32_t bytes,
								    const uint64_t compare_add,
								    const uint64_t swap_add)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_raddr_seg rseg;
	struct mlx5_wqe_atomic_seg atseg;
	struct mlx5_wqe_data_seg dseg;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
	cseg.fm_ce_se = ctrl_flags;

	rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rseg.rkey = rkey;
#else
	rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

	atseg.swap_add = doca_gpu_dev_verbs_bswap64(opcode == MLX5_OPCODE_ATOMIC_FA ? compare_add : swap_add);
	atseg.compare = doca_gpu_dev_verbs_bswap64(compare_add);

	dseg.byte_count = doca_gpu_dev_verbs_bswap32(bytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg.lkey = lkey;
#else
	dseg.lkey = doca_gpu_dev_verbs_bswap32(lkey);
#endif
	dseg.addr = doca_gpu_dev_verbs_bswap64(laddr);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(atseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg));
}

/**
 * @brief Prepare a Wait WQE
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ static inline void doca_gpu_dev_verbs_wqe_prepare_wait(struct doca_gpu_dev_verbs_qp *qp,
								  struct doca_gpu_dev_verbs_wqe *wqe_ptr,
								  uint16_t wqe_idx,
								  enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
								  const uint32_t max_index,
								  const uint32_t qpn_cqn)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct doca_gpu_dev_verbs_wqe_wait_seg wseg;

	cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
							   DOCA_GPUNETIO_MLX5_OPCODE_WAIT);
	cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_WAIT);
	cseg.fm_ce_se = ctrl_flags;
	// cseg.imm = 0;

	wseg.max_index = doca_gpu_dev_verbs_bswap32(max_index);
	wseg.qpn_cqn = doca_gpu_dev_verbs_bswap32(qpn_cqn);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(wseg));
}

/**
 * @brief Prepare an Send WQE with 0 data segment
 * Typical usage: send with immediate without carrying data.
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Pointer to the WQE buffer to send the prepared WQE to
 * @param immediate - Immediate value to use in case of DOCA_GPUNETIO_MLX5_OPCODE_SEND_IMM opcode
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_send(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							   const uint16_t wqe_idx,
							   const uint32_t opcode,
							   enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							   const uint32_t immediate)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_1ds);
	cseg.fm_ce_se = ctrl_flags;
	cseg.imm = immediate;

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
}

/**
 * @brief Prepare an Send WQE with 1 data segment
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Pointer to the WQE buffer to send the prepared WQE to
 * @param immediate - Immediate value to use in case of DOCA_GPUNETIO_MLX5_OPCODE_SEND_IMM opcode
 * @param laddr0 - Local address to send with first data segment
 * @param lkey0 - Local address mkey to send with first data segment
 * @param bytes0 - Local address size to send with first data segment
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_send(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							   const uint16_t wqe_idx,
							   const uint32_t opcode,
							   enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							   const uint32_t immediate,
							   const uint64_t laddr0,
							   const uint32_t lkey0,
							   const uint32_t bytes0)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_data_seg dseg0;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_2ds);
	cseg.fm_ce_se = ctrl_flags;
	cseg.imm = immediate;

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg0));
}

/**
 * @brief Prepare an Send WQE with 2 data segments
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Pointer to the WQE buffer to send the prepared WQE to
 * @param immediate - Immediate value to use in case of DOCA_GPUNETIO_MLX5_OPCODE_SEND_IMM opcode
 * @param laddr0 - Local address to send with first data segment
 * @param lkey0 - Local address mkey to send with first data segment
 * @param bytes0 - Local address size to send with first data segment
 * @param laddr1 - Local address to send with second data segment
 * @param lkey1 - Local address mkey to send with second data segment
 * @param bytes1 - Local address size to send with second data segment
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_send(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							   const uint16_t wqe_idx,
							   const uint32_t opcode,
							   enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							   const uint32_t immediate,
							   const uint64_t laddr0,
							   const uint32_t lkey0,
							   const uint32_t bytes0,
							   const uint64_t laddr1,
							   const uint32_t lkey1,
							   const uint32_t bytes1)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_data_seg dseg0;
	struct mlx5_wqe_data_seg dseg1;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_3ds);
	cseg.fm_ce_se = ctrl_flags;
	cseg.imm = immediate;

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	dseg1.byte_count = doca_gpu_dev_verbs_bswap32(bytes1);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg1.lkey = lkey1;
#else
	dseg1.lkey = doca_gpu_dev_verbs_bswap32(lkey1);
#endif
	dseg1.addr = doca_gpu_dev_verbs_bswap64(laddr1);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg0));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg1));
}

/**
 * @brief Prepare an Send WQE with 3 data segments
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Send Work Request pointer to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Pointer to the WQE buffer to send the prepared WQE to
 * @param immediate - Immediate value to use in case of DOCA_GPUNETIO_MLX5_OPCODE_SEND_IMM opcode
 * @param laddr0 - Local address to send with first data segment
 * @param lkey0 - Local address mkey to send with first data segment
 * @param bytes0 - Local address size to send with first data segment
 * @param laddr1 - Local address to send with second data segment
 * @param lkey1 - Local address mkey to send with second data segment
 * @param bytes1 - Local address size to send with second data segment
 * @param laddr2 - Local address to send with third data segment
 * @param lkey2 - Local address mkey to send with third data segment
 * @param bytes2 - Local address size to send with third data segment
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_send(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_wqe *wqe_ptr,
							   const uint16_t wqe_idx,
							   const uint32_t opcode,
							   enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
							   const uint32_t immediate,
							   const uint64_t laddr0,
							   const uint32_t lkey0,
							   const uint32_t bytes0,
							   const uint64_t laddr1,
							   const uint32_t lkey1,
							   const uint32_t bytes1,
							   const uint64_t laddr2,
							   const uint32_t lkey2,
							   const uint32_t bytes2)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_data_seg dseg0;
	struct mlx5_wqe_data_seg dseg1;
	struct mlx5_wqe_data_seg dseg2;

	cseg.opmod_idx_opcode =
		doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
	cseg.fm_ce_se = ctrl_flags;
	cseg.imm = immediate;

	dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg0.lkey = lkey0;
#else
	dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
	dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

	dseg1.byte_count = doca_gpu_dev_verbs_bswap32(bytes1);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg1.lkey = lkey1;
#else
	dseg1.lkey = doca_gpu_dev_verbs_bswap32(lkey1);
#endif
	dseg1.addr = doca_gpu_dev_verbs_bswap64(laddr1);

	dseg2.byte_count = doca_gpu_dev_verbs_bswap32(bytes2);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg2.lkey = lkey2;
#else
	dseg2.lkey = doca_gpu_dev_verbs_bswap32(lkey2);
#endif
	dseg2.addr = doca_gpu_dev_verbs_bswap64(laddr2);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg0));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg1));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg2));
}

/**
 * @brief Prepare an Recv WQE
 *
 * @param qp - Queue Pair (QP)
 * @param rwqe - Recv Work Request pointer to be prepared
 * @param addr - Local address
 * @param mkey - Local address mkey
 * @param bytes - Local address size
 */
__device__ inline void doca_gpu_dev_verbs_wqe_prepare_recv(struct doca_gpu_dev_verbs_qp *qp,
							   struct mlx5_wqe_data_seg *rwqe,
							   const uint64_t addr,
							   const uint32_t mkey,
							   const uint32_t bytes)
{
	struct mlx5_wqe_data_seg rwqe_;

	rwqe_.byte_count = doca_gpu_dev_verbs_bswap32(bytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	rwqe_.lkey = mkey;
#else
	rwqe_.lkey = doca_gpu_dev_verbs_bswap32(mkey);
#endif
	rwqe_.addr = doca_gpu_dev_verbs_bswap64(addr);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)rwqe, (uint64_t *)&(rwqe_));
}

/**
 * @brief Prepare a Dump WQE
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Pointer to the WQE buffer to write the prepared WQE to
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Control flags for the WQE
 * @param laddr - Local address for local dump
 * @param lkey - Local address mkey for local dump
 * @param bytes - Local address bytes to dump
 */
__device__ static inline void doca_gpu_dev_verbs_wqe_prepare_dump(struct doca_gpu_dev_verbs_qp *qp,
								  struct doca_gpu_dev_verbs_wqe *wqe_ptr,
								  const uint16_t wqe_idx,
								  enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
								  const uint64_t laddr,
								  const uint32_t lkey,
								  const uint32_t bytes)
{
	struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
	struct mlx5_wqe_data_seg dseg;

	cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
							   DOCA_GPUNETIO_MLX5_OPCODE_DUMP);
	cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_2ds);
	cseg.fm_ce_se = ctrl_flags;

	dseg.byte_count = doca_gpu_dev_verbs_bswap32(bytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
	dseg.lkey = lkey;
#else
	dseg.lkey = doca_gpu_dev_verbs_bswap32(lkey);
#endif
	dseg.addr = doca_gpu_dev_verbs_bswap64(laddr);

	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
	doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg));
}

/* **************************************** WAIT **************************************** */
/**
 * @brief Poll a specific CQE in the QP SQ
 *
 * @param qp - Queue Pair (QP)
 * @param ticket - CQE index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_wait(struct doca_gpu_dev_verbs_qp *qp,
						      doca_gpu_dev_verbs_ticket_t *ticket)
{
	doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), *ticket);
}

/**
 * @brief Poll the CQE corresponding to the latest WQE posted in the SQ
 *
 * @param qp - Queue Pair (QP)
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_wait(struct doca_gpu_dev_verbs_qp *qp)
{
	uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qp->sq_rsvd_index);
	[[unlikely]] if (ticket == 0)
		return;
	--ticket;
	doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), ticket);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_QP_H */

/** @} */
