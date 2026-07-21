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

#ifndef DOCA_GPUNETIO_DEV_VERBS_TWOSIDED_CUH
#define DOCA_GPUNETIO_DEV_VERBS_TWOSIDED_CUH

#include <doca_gpunetio_dev_verbs_qp.cuh>
#include <doca_gpunetio_dev_verbs_cq.cuh>

/* **************************************** SEND **************************************** */
/**
 * @brief Send shared QP function. Every thread posts a different RDMA Send in a different WQE position.
 * It assumes RDMA Sends with a single data segment.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] laddr - RDMA Send local info
 * @param[in] size - RDMA Send size (bytes)
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_send_thread(struct doca_gpu_dev_verbs_qp *qp,
							     struct doca_gpu_dev_verbs_addr laddr,
							     size_t size,
							     doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx = 0;
	uint64_t wqe_idx;
	size_t remaining_size = size;
	size_t size_;
	uint64_t num_chunks =
		doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

	base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks);
#pragma unroll 1
	for (uint64_t i = 0; i < num_chunks; i++) {
		wqe_idx = base_wqe_idx + i;
		size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE :
										 remaining_size;
		wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

		doca_gpu_dev_verbs_wqe_prepare_send(qp,
						    wqe_ptr,
						    wqe_idx,
						    DOCA_GPUNETIO_MLX5_OPCODE_SEND,
						    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
						    0,
						    laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE),
						    laddr.key,
						    size_);
		remaining_size -= size_;
	}

	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
	doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp,
													  wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Send shared QP function. Every thread in the warp posts a different RDMA Send in a different WQE position.
 * It assumes RDMA Sends with a single data segment.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] laddr - RDMA Send local info
 * @param[in] size - RDMA Send size (bytes)
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_send_warp(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_addr laddr,
							   size_t size,
							   doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx = 0;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t wqe_idx;
	uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();

	DOCA_GPUNETIO_VERBS_ASSERT(size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	if (lane_idx == 0) {
		base_wqe_idx =
			doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, DOCA_GPUNETIO_VERBS_WARP_SIZE);
		base_wqe_idx_0 = (uint32_t)base_wqe_idx;
		base_wqe_idx_1 = (uint32_t)(base_wqe_idx >> 32);
	}
	__syncwarp();

	base_wqe_idx_0 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_0);
	base_wqe_idx_1 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_1);
	base_wqe_idx = ((uint64_t)base_wqe_idx_1) << 32 | base_wqe_idx_0;

	wqe_idx = base_wqe_idx + lane_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

	doca_gpu_dev_verbs_wqe_prepare_send(qp,
					    wqe_ptr,
					    wqe_idx,
					    DOCA_GPUNETIO_MLX5_OPCODE_SEND,
					    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
					    0,
					    laddr.addr,
					    laddr.key,
					    size);

	__syncwarp();
	if (lane_idx == 0) {
		doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp,
									  base_wqe_idx,
									  base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE - 1);
		doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
			qp,
			base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE);
	}
	__syncwarp();

	*out_ticket = wqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Send shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] laddr - RDMA Send local info
 * @param[in] size - RDMA Send size (bytes)
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_send(struct doca_gpu_dev_verbs_qp *qp,
						      struct doca_gpu_dev_verbs_addr laddr,
						      size_t size,
						      doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_send_thread<resource_sharing_mode, nic_handler>(qp, laddr, size, out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_send_warp<resource_sharing_mode, nic_handler>(qp, laddr, size, out_ticket);
}

/* **************************************** RECV **************************************** */
/**
 * @brief Recv shared QP function. Every thread posts a different RDMA Recv in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] laddr - RDMA Recv local info
 * @param[in] size - RDMA Recv size (bytes)
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_recv_thread(struct doca_gpu_dev_verbs_qp *qp,
							     struct doca_gpu_dev_verbs_addr laddr,
							     size_t size,
							     doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	uint64_t rwqe_idx;
	struct mlx5_wqe_data_seg *rwqe_ptr;

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	rwqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, 1);
	rwqe_ptr = doca_gpu_dev_verbs_get_rwqe_ptr(qp, rwqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_recv(qp, rwqe_ptr, laddr.addr, laddr.key, size);
	doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rwqe_idx, rwqe_idx);
	doca_gpu_dev_verbs_submit<resource_sharing_mode,
				DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
				nic_handler,
				DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rwqe_idx + 1);

	*out_ticket = rwqe_idx;
}

/**
 * @brief Recv shared QP function. Every thread in the warp posts a different RDMA Recv in a different WQE position.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] laddr - RDMA Recv local info
 * @param[in] size - RDMA Recv size (bytes)
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_recv_warp(struct doca_gpu_dev_verbs_qp *qp,
							   struct doca_gpu_dev_verbs_addr laddr,
							   size_t size,
							   doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct mlx5_wqe_data_seg *rwqe_ptr;
	uint64_t base_wqe_idx = 0;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t rwqe_idx;
	uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();

	DOCA_GPUNETIO_VERBS_ASSERT(size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	if (lane_idx == 0) {
		base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_RQ>(
				qp,
				DOCA_GPUNETIO_VERBS_WARP_SIZE);

		base_wqe_idx_0 = (uint32_t)base_wqe_idx;
		base_wqe_idx_1 = (uint32_t)(base_wqe_idx >> 32);
	}
	__syncwarp();

	base_wqe_idx_0 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_0);
	base_wqe_idx_1 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_1);
	base_wqe_idx = ((uint64_t)base_wqe_idx_1) << 32 | base_wqe_idx_0;

	rwqe_idx = base_wqe_idx + lane_idx;
	rwqe_ptr = doca_gpu_dev_verbs_get_rwqe_ptr(qp, rwqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_recv(qp, rwqe_ptr, laddr.addr, laddr.key, size);
	__syncwarp();

	if (lane_idx == 0) {
		doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_RQ>(
						qp,
						base_wqe_idx,
						base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE - 1);
		doca_gpu_dev_verbs_submit<resource_sharing_mode,
						DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
						nic_handler,
						DOCA_GPUNETIO_VERBS_QP_RQ>(qp, base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE);

	}
	__syncwarp();

	*out_ticket = rwqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Recv QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] laddr - RDMA Recv local info
 * @param[in] size - RDMA Recv size (bytes)
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_recv(struct doca_gpu_dev_verbs_qp *qp,
						      struct doca_gpu_dev_verbs_addr raddr,
						      size_t size,
						      doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_recv_thread<resource_sharing_mode, nic_handler>(qp, raddr, size, out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_recv_warp<resource_sharing_mode, nic_handler>(qp, raddr, size, out_ticket);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED,
		  enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED>
__device__ static inline void doca_priv_gpu_dev_verbs_recv_wait(struct doca_gpu_dev_verbs_qp *qp,
                                                            struct doca_gpu_dev_verbs_addr daddr,
                                                            doca_gpu_dev_verbs_ticket_t *ticket,
															struct mlx5_cqe64 **cqe64) {
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t wqe_idx = 0;

    if (mcst_mode == DOCA_GPUNETIO_VERBS_MCST_ENABLED) {
        wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1);
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        doca_gpu_dev_verbs_wqe_prepare_dump(qp, wqe_ptr, wqe_idx,
                                            DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
											daddr.addr, daddr.key, 1);

        doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);
        doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                  nic_handler>(qp, wqe_idx + 1);

		doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), wqe_idx);
		doca_gpu_dev_verbs_poll_cq<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_RQ, blocking_mode>(doca_gpu_dev_verbs_qp_get_cq_rq(qp), *ticket, cqe64);
    } else {
		doca_gpu_dev_verbs_poll_cq<resource_sharing_mode, DOCA_GPUNETIO_VERBS_QP_RQ, blocking_mode>(doca_gpu_dev_verbs_qp_get_cq_rq(qp), *ticket, cqe64);
	}
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED,
		  enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED>
__device__ static inline void doca_gpu_dev_verbs_recv_wait(struct doca_gpu_dev_verbs_qp *qp,
                                                            struct doca_gpu_dev_verbs_addr daddr,
															doca_gpu_dev_verbs_ticket_t *ticket,
															struct mlx5_cqe64 **cqe64) {
	doca_priv_gpu_dev_verbs_recv_wait<resource_sharing_mode, nic_handler, mcst_mode, blocking_mode>(qp, daddr, ticket, cqe64);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED,
		  enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED>
__device__ static inline void doca_gpu_dev_verbs_recv_wait(struct doca_gpu_dev_verbs_qp *qp,
                                                            struct doca_gpu_dev_verbs_addr daddr,
															struct mlx5_cqe64 **cqe64) {
	uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qp->rq_rsvd_index);
	[[unlikely]] if (ticket == 0)
		return;
	--ticket;

	doca_priv_gpu_dev_verbs_recv_wait<resource_sharing_mode, nic_handler, mcst_mode, blocking_mode>(qp, daddr, &ticket, cqe64);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_TWOSIDED_CUH */
