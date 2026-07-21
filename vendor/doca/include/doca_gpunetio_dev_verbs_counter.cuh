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
 * @file doca_gpunetio_dev_verbs_counter.cuh
 * @page DOCA GPUNetIO CUDA Device functions with src exposed
 * @defgroup DOCA_GPUNETIO_DEV_DEF DOCA GPUNetIO Device - Counter Shared QP ops
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */

#ifndef DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH
#define DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH

#include <doca_gpunetio_dev_verbs_qp.cuh>
#include <doca_gpunetio_dev_verbs_cq.cuh>

/**
 * @brief Submit work requests to the NIC using the DB protocol for multiple QPs.
 * Typically used with counter operations where both main and companion qp must be updated.
 *
 * @param qps - Array of Queue Pair (QP)
 * @param prod_indices - Array of producer indices
 * @param num_qps - Number of Queue Pair (QP)
 */
template <unsigned int num_qps,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_db_multi_qps(struct doca_gpu_dev_verbs_qp **qps,
									      uint64_t *prod_indices)
{
	DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
	uint64_t old_prod_indices[num_qps];
	__be64 db_vals[num_qps];

#pragma unroll 2
	for (unsigned int i = 0; i < num_qps; i++) {
		doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qps[i]->sq_lock);
		old_prod_indices[i] =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qps[i]->sq_wqe_pi, prod_indices[i]);
		if (old_prod_indices[i] < prod_indices[i]) {
			// Early rining of the DB to push WQEs to the NIC ASAP.
			__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
			db_vals[i] = doca_gpu_dev_verbs_prepare_db(qps[i], prod_indices[i]);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
			if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
				doca_gpu_dev_verbs_async_store_release<sync_scope>((uint64_t *)db_ptr, (uint64_t)db_vals[i]);
			} else
#endif
			{
				doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
				{
					doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_vals[i]);
				}
#else
				{
					cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
					db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
				}
#endif
			}
		}
	}

#pragma unroll 2
	for (unsigned int i = 0; i < num_qps; i++) {
		if (old_prod_indices[i] < prod_indices[i]) {
			// In case the recovery path is triggered, the later DB ringing will cover for
			// correctness.
			doca_priv_gpu_dev_verbs_update_dbr(qps[i], prod_indices[i]);
			__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
			if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
				doca_gpu_dev_verbs_async_store_release<sync_scope>((uint64_t *)db_ptr, (uint64_t)db_vals[i]);
			} else
#endif
			{
				doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
				{
					doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_vals[i]);
				}
#else
				{
					cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
					db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
				}
#endif
			}
		}
		doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qps[i]->sq_lock);
	}
}

/**
 * @brief Submit work requests to the NIC using the DB protocol for multiple QPs with send DBR mode set to external.
 * Typically used with counter operations where both main and companion qp must be updated.
 *
 * @param qps - Array of Queue Pair (QP)
 * @param prod_indices - Array of producer indices
 * @param num_qps - Number of Queue Pair (QP)
 */
template <unsigned int num_qps,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_gpu_code_opt code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr(struct doca_gpu_dev_verbs_qp **qps,
										      uint64_t *prod_indices)
{
	DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
	uint64_t old_prod_indices[num_qps];
	__be64 db_vals[num_qps];

#pragma unroll 2
	for (unsigned int i = 0; i < num_qps; i++) {
		old_prod_indices[i] =
			doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qps[i]->sq_wqe_pi, prod_indices[i]);
		if (old_prod_indices[i] < prod_indices[i]) {
			// Early rining of the DB to push WQEs to the NIC ASAP.
			__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
			db_vals[i] = doca_gpu_dev_verbs_prepare_db(qps[i], prod_indices[i]);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
			if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
				doca_gpu_dev_verbs_async_store_release<sync_scope>((uint64_t *)db_ptr, (uint64_t)db_vals[i]);
			} else
#endif
			{
				doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
				{
					doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_vals[i]);
				}
#else
				{
					cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
					db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
				}
#endif
			}
		}
	}
}

/**
 * @brief Submit work requests to the NIC using the CPU proxy protocol for multiple QPs.
 * Typically used with counter operations where both main and companion qp must be updated.
 *
 * @param qps - Array of Queue Pair (QP)
 * @param prod_indices - Array of producer indices
 * @param num_qps - Number of Queue Pair (QP)
 */
template <unsigned int num_qps,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_verbs_submit_proxy_multi_qps(struct doca_gpu_dev_verbs_qp **qps,
									uint64_t *prod_indices)
{
	DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
	doca_gpu_dev_verbs_fence_release<sync_scope>();

#pragma unroll 2
	for (unsigned int i = 0; i < num_qps; i++) {
		doca_gpu_dev_verbs_ring_proxy<resource_sharing_mode>(qps[i], prod_indices[i]);
	}
}

/**
 * @brief Chose the best method to submit work requests to the NIC for multiple QPs.
 * Typically used with counter operations where both main and companion qp must be updated.
 *
 * @param qps - Array of Queue Pair (QP)
 * @param prod_indices - Array of producer indices
 * @param num_qps - Number of Queue Pair (QP)
 */
template <unsigned int num_qps,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_submit_multi_qps(struct doca_gpu_dev_verbs_qp **qps,
								  uint64_t *prod_indices)
{
	DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
	if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
		const enum doca_gpu_dev_verbs_nic_handler qp_nic_handler =
			(enum doca_gpu_dev_verbs_nic_handler)__ldg((int *)&qps[0]->nic_handler);
		if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
			doca_gpu_dev_verbs_submit_db_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
				qps,
				prod_indices);
		else if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
			doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr<num_qps, resource_sharing_mode, sync_scope>(
				qps,
				prod_indices);
		else
			doca_gpu_dev_verbs_submit_proxy_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
				qps,
				prod_indices);
	} else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
		doca_gpu_dev_verbs_submit_db_multi_qps<num_qps, resource_sharing_mode, sync_scope>(qps, prod_indices);
	else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
		doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr<num_qps, resource_sharing_mode, sync_scope>(
			qps,
			prod_indices);
	else
		doca_gpu_dev_verbs_submit_proxy_multi_qps<num_qps, resource_sharing_mode, sync_scope>(qps,
												      prod_indices);
}

/* **************************************** PUT COUNTER **************************************** */
/**
 * @brief Put + counter shared QP function. Every thread posts a different RDMA Write on the main QP.
 * Then, every thread posts on the companion QP two WQEs:
 * - WQE Wait on the CQE index corresponding to the RDMA Write posted in the main QP.
 * - RDMA Atomic FetchAdd that will be executed as soon as the CQE arrives in the main SQ CQ (i.e. RDMA Write has been
 * executed). Every thread submits the WQEs.
 *
 * @param[in] qp - Main Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] companion_qp - Companion Queue Pair (QP)
 * @param[in] counter_raddr - RDMA Atomic remote info
 * @param[in] counter_laddr - RDMA Atomic local info
 * @param[in] counter_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_counter_thread(struct doca_gpu_dev_verbs_qp *qp,
								    struct doca_gpu_dev_verbs_addr raddr,
								    struct doca_gpu_dev_verbs_addr laddr,
								    size_t size,
								    struct doca_gpu_dev_verbs_qp *companion_qp,
								    struct doca_gpu_dev_verbs_addr counter_raddr,
								    struct doca_gpu_dev_verbs_addr counter_laddr,
								    uint64_t counter_val)
{
	constexpr unsigned int num_qps = 2;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx;
	uint64_t wqe_idx;
	size_t remaining_size = size;
	size_t size_;
	uint64_t num_chunks =
		doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);

	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(companion_qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);
	DOCA_GPUNETIO_VERBS_ASSERT(companion_qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

	base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks);
#pragma unroll 1
	for (uint64_t i = 0; i < num_chunks; i++) {
		wqe_idx = base_wqe_idx + i;
		size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE :
										 remaining_size;
		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

		doca_gpu_dev_verbs_wqe_prepare_write(qp,
						     wqe_ptr,
						     wqe_idx,
						     DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
						     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
						     0,
						     raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
						     raddr.key,
						     laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
						     laddr.key,
						     size_);
		remaining_size -= size_;
	}

	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

	uint64_t companion_base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
	uint64_t companion_wqe_idx = companion_base_wqe_idx;

	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp,
					    wqe_ptr,
					    companion_wqe_idx,
					    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					    wqe_idx,
					    qp->cq_sq.cq_num);

	++companion_wqe_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_atomic(companion_qp,
					      wqe_ptr,
					      companion_wqe_idx,
					      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
					      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					      counter_raddr.addr,
					      counter_raddr.key,
					      counter_laddr.addr,
					      counter_laddr.key,
					      sizeof(uint64_t),
					      counter_val,
					      0);
	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp,
								  companion_base_wqe_idx,
								  companion_wqe_idx);

	doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
	uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
	doca_gpu_dev_verbs_submit_multi_qps<num_qps,
					    resource_sharing_mode,
					    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
					    nic_handler>(qps, prod_indices);
}

/**
 * @brief Put + counter shared QP function. Every thread posts a different RDMA Write on the main QP.
 * Then, only the thread with lane_idx = 0 posts on the companion QP two WQEs:
 * - WQE Wait on the CQE index corresponding to the RDMA Write posted in the main QP.
 * - RDMA Atomic FetchAdd that will be executed as soon as the CQE arrives in the main SQ CQ (i.e. RDMA Write has been
 * executed). Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param[in] qp - Main Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] companion_qp - Companion Queue Pair (QP)
 * @param[in] counter_raddr - RDMA Atomic remote info
 * @param[in] counter_laddr - RDMA Atomic local info
 * @param[in] counter_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_counter_warp(struct doca_gpu_dev_verbs_qp *qp,
								  struct doca_gpu_dev_verbs_addr raddr,
								  struct doca_gpu_dev_verbs_addr laddr,
								  size_t size,
								  struct doca_gpu_dev_verbs_qp *companion_qp,
								  struct doca_gpu_dev_verbs_addr counter_raddr,
								  struct doca_gpu_dev_verbs_addr counter_laddr,
								  uint64_t counter_val)
{
#if __CUDA_ARCH__ >= 800
	constexpr unsigned int num_qps = 2;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx = 0;
	uint64_t wqe_idx;
	uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();

	DOCA_GPUNETIO_VERBS_ASSERT(size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(companion_qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);
	DOCA_GPUNETIO_VERBS_ASSERT(companion_qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

	if (lane_idx == 0)
		base_wqe_idx =
			doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, DOCA_GPUNETIO_VERBS_WARP_SIZE);
	__syncwarp();

	base_wqe_idx = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, (uint32_t)base_wqe_idx);

	wqe_idx = base_wqe_idx + lane_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_write(qp,
					     wqe_ptr,
					     wqe_idx,
					     DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
					     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					     0,
					     raddr.addr,
					     raddr.key,
					     laddr.addr,
					     laddr.key,
					     size);

	__syncwarp();
	if (lane_idx == 0) {
		wqe_idx = base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE - 1;
		doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

		uint64_t companion_base_wqe_idx =
			doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
		uint64_t companion_wqe_idx = companion_base_wqe_idx;

		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
		doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp,
						    wqe_ptr,
						    companion_wqe_idx,
						    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
						    wqe_idx,
						    qp->cq_sq.cq_num);

		++companion_wqe_idx;
		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
		doca_gpu_dev_verbs_wqe_prepare_atomic(companion_qp,
						      wqe_ptr,
						      companion_wqe_idx,
						      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
						      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
						      counter_raddr.addr,
						      counter_raddr.key,
						      counter_laddr.addr,
						      counter_laddr.key,
						      sizeof(uint64_t),
						      counter_val,
						      0);
		doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp,
									  companion_base_wqe_idx,
									  companion_wqe_idx);

		doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
		uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
		doca_gpu_dev_verbs_submit_multi_qps<num_qps,
						    resource_sharing_mode,
						    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
						    nic_handler>(qps, prod_indices);
	}
	__syncwarp();
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
#endif
}

/**
 * @brief Put + counter shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Main Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] companion_qp - Companion Queue Pair (QP)
 * @param[in] counter_raddr - RDMA Atomic remote info
 * @param[in] counter_laddr - RDMA Atomic local info
 * @param[in] counter_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_put_counter(struct doca_gpu_dev_verbs_qp *qp,
							     struct doca_gpu_dev_verbs_addr raddr,
							     struct doca_gpu_dev_verbs_addr laddr,
							     size_t size,
							     struct doca_gpu_dev_verbs_qp *companion_qp,
							     struct doca_gpu_dev_verbs_addr counter_raddr,
							     struct doca_gpu_dev_verbs_addr counter_laddr,
							     uint64_t counter_val)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_put_counter_thread<resource_sharing_mode, nic_handler>(qp,
											  raddr,
											  laddr,
											  size,
											  companion_qp,
											  counter_raddr,
											  counter_laddr,
											  counter_val);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_put_counter_warp<resource_sharing_mode, nic_handler>(qp,
											raddr,
											laddr,
											size,
											companion_qp,
											counter_raddr,
											counter_laddr,
											counter_val);
}

/**
 * @brief Put inline + counter shared QP function. Every thread posts a different RDMA Write Inline on the main QP.
 * Then, every thread posts on the companion QP two WQEs:
 * - WQE Wait on the CQE index corresponding to the RDMA Write posted in the main QP.
 * - RDMA Atomic FetchAdd that will be executed as soon as the CQE arrives in the main SQ CQ (i.e. RDMA Write has been
 * executed). Every thread submits the WQEs.
 *
 * @param[in] qp - Main Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] companion_qp - Companion Queue Pair (QP)
 * @param[in] counter_raddr - RDMA Atomic remote info
 * @param[in] counter_laddr - RDMA Atomic local info
 * @param[in] counter_val - RDMA Atomic value
 */
template <typename T,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_counter(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_addr raddr,
							   T value,
							   struct doca_gpu_dev_verbs_qp *companion_qp,
							   struct doca_gpu_dev_verbs_addr counter_raddr,
							   struct doca_gpu_dev_verbs_addr counter_laddr,
							   uint64_t counter_val)
{
	constexpr unsigned int num_qps = 2;
	uint64_t wqe_idx;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;

	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

	wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1);
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp,
							     wqe_ptr,
							     wqe_idx,
							     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
							     raddr.addr,
							     raddr.key,
							     sizeof(T));
	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<T>(qp, wqe_ptr, value);
	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);

	uint64_t companion_base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
	uint64_t companion_wqe_idx = companion_base_wqe_idx;

	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp,
					    wqe_ptr,
					    companion_wqe_idx,
					    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					    wqe_idx,
					    qp->cq_sq.cq_num);

	++companion_wqe_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_atomic(companion_qp,
					      wqe_ptr,
					      companion_wqe_idx,
					      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
					      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					      counter_raddr.addr,
					      counter_raddr.key,
					      counter_laddr.addr,
					      counter_laddr.key,
					      sizeof(uint64_t),
					      counter_val,
					      0);
	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp,
								  companion_base_wqe_idx,
								  companion_wqe_idx);

	doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
	uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
	doca_gpu_dev_verbs_submit_multi_qps<num_qps,
					    resource_sharing_mode,
					    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
					    nic_handler>(qps, prod_indices);
}

/**
 * @brief Put + signal + counter shared QP function. Every thread posts a different RDMA Write + RDMA Atomic FetchAdd on
 * the main QP. Then, every thread posts on the companion QP two WQEs:
 * - WQE Wait on the CQE index corresponding to the RDMA Atomic posted in the main QP.
 * - RDMA Atomic FetchAdd that will be executed as soon as the CQE arrives in the main SQ CQ (i.e. RDMA Atomic has been
 * executed). Every thread submits the WQEs.
 *
 * @param[in] qp - Main Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] companion_qp - Companion Queue Pair (QP)
 * @param[in] counter_raddr - RDMA Atomic remote info
 * @param[in] counter_laddr - RDMA Atomic local info
 * @param[in] counter_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_signal_counter(struct doca_gpu_dev_verbs_qp *qp,
								    struct doca_gpu_dev_verbs_addr raddr,
								    struct doca_gpu_dev_verbs_addr laddr,
								    size_t size,
								    struct doca_gpu_dev_verbs_addr sig_raddr,
								    struct doca_gpu_dev_verbs_addr sig_laddr,
								    uint64_t sig_val,
								    struct doca_gpu_dev_verbs_qp *companion_qp,
								    struct doca_gpu_dev_verbs_addr counter_raddr,
								    struct doca_gpu_dev_verbs_addr counter_laddr,
								    uint64_t counter_val)
{
	constexpr unsigned int num_qps = 2;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx;
	uint64_t wqe_idx;
	size_t remaining_size = size;
	size_t size_;
	uint64_t num_chunks =
		doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);

	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

	// Put
	base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks + 1);
#pragma unroll 1
	for (uint64_t i = 0; i < num_chunks; i++) {
		wqe_idx = base_wqe_idx + i;
		size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE :
										 remaining_size;
		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

		doca_gpu_dev_verbs_wqe_prepare_write(qp,
						     wqe_ptr,
						     wqe_idx,
						     DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
						     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
						     0,
						     raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
						     raddr.key,
						     laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
						     laddr.key,
						     size_);
		remaining_size -= size_;
	}

	// Signal
	++wqe_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_atomic(qp,
					      wqe_ptr,
					      wqe_idx,
					      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
					      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					      sig_raddr.addr,
					      sig_raddr.key,
					      sig_laddr.addr,
					      sig_laddr.key,
					      sizeof(uint64_t),
					      sig_val,
					      0);

	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

	// Counter
	uint64_t companion_base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
	uint64_t companion_wqe_idx = companion_base_wqe_idx;

	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp,
					    wqe_ptr,
					    companion_wqe_idx,
					    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					    wqe_idx,
					    qp->cq_sq.cq_num);

	++companion_wqe_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_atomic(companion_qp,
					      wqe_ptr,
					      companion_wqe_idx,
					      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
					      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					      counter_raddr.addr,
					      counter_raddr.key,
					      counter_laddr.addr,
					      counter_laddr.key,
					      sizeof(uint64_t),
					      counter_val,
					      0);
	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp,
								  companion_base_wqe_idx,
								  companion_wqe_idx);

	doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
	uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
	doca_gpu_dev_verbs_submit_multi_qps<num_qps,
					    resource_sharing_mode,
					    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
					    nic_handler>(qps, prod_indices);
}

/**
 * @brief Signal + counter shared QP function. Every thread posts a different RDMA Atomic FetchAdd on the main QP.
 * Then, every thread posts on the companion QP two WQEs:
 * - WQE Wait on the CQE index corresponding to the RDMA Atomic posted in the main QP.
 * - RDMA Atomic FetchAdd that will be executed as soon as the CQE arrives in the main SQ CQ (i.e. RDMA Atomic has been
 * executed). Every thread submits the WQEs.
 *
 * @param[in] qp - Main Queue Pair (QP)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] companion_qp - Companion Queue Pair (QP)
 * @param[in] counter_raddr - RDMA Atomic remote info
 * @param[in] counter_laddr - RDMA Atomic local info
 * @param[in] counter_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_signal_counter(struct doca_gpu_dev_verbs_qp *qp,
								struct doca_gpu_dev_verbs_addr sig_raddr,
								struct doca_gpu_dev_verbs_addr sig_laddr,
								uint64_t sig_val,
								struct doca_gpu_dev_verbs_qp *companion_qp,
								struct doca_gpu_dev_verbs_addr counter_raddr,
								struct doca_gpu_dev_verbs_addr counter_laddr,
								uint64_t counter_val)
{
	constexpr unsigned int num_qps = 2;
	uint64_t wqe_idx;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;

	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

	// Signal
	wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1);
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_atomic(qp,
					      wqe_ptr,
					      wqe_idx,
					      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
					      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					      sig_raddr.addr,
					      sig_raddr.key,
					      sig_laddr.addr,
					      sig_laddr.key,
					      sizeof(uint64_t),
					      sig_val,
					      0);

	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);

	// Counter
	uint64_t companion_base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2);
	uint64_t companion_wqe_idx = companion_base_wqe_idx;

	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp,
					    wqe_ptr,
					    companion_wqe_idx,
					    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					    wqe_idx,
					    qp->cq_sq.cq_num);

	++companion_wqe_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_atomic(companion_qp,
					      wqe_ptr,
					      companion_wqe_idx,
					      DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA,
					      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					      counter_raddr.addr,
					      counter_raddr.key,
					      counter_laddr.addr,
					      counter_laddr.key,
					      sizeof(uint64_t),
					      counter_val,
					      0);
	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp,
								  companion_base_wqe_idx,
								  companion_wqe_idx);

	doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
	uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};
	doca_gpu_dev_verbs_submit_multi_qps<num_qps,
					    resource_sharing_mode,
					    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
					    nic_handler>(qps, prod_indices);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH */
