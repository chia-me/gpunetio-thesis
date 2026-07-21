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
 * @file doca_gpunetio_dev_verbs_onesided.cuh
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

#ifndef DOCA_GPUNETIO_DEV_VERBS_ONESIDED_CUH
#define DOCA_GPUNETIO_DEV_VERBS_ONESIDED_CUH

#include <doca_gpunetio_dev_verbs_qp.cuh>
#include <doca_gpunetio_dev_verbs_cq.cuh>

/* **************************************** PUT **************************************** */
/**
 * @brief Put shared QP function. Every thread posts a different RDMA Write in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_thread(struct doca_gpu_dev_verbs_qp *qp,
							    struct doca_gpu_dev_verbs_addr raddr,
							    struct doca_gpu_dev_verbs_addr laddr,
							    size_t size,
							    doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx;
	uint64_t wqe_idx;
	size_t remaining_size = size;
	size_t size_;
	uint64_t num_chunks =
		doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

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
	doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp,
													  wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Put shared QP function. Every thread in the warp posts a different RDMA Write in a different WQE position.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_warp(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  struct doca_gpu_dev_verbs_addr laddr,
							  size_t size,
							  doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t base_wqe_idx = 0;
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
 * @brief Put shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_put(struct doca_gpu_dev_verbs_qp *qp,
						     struct doca_gpu_dev_verbs_addr raddr,
						     struct doca_gpu_dev_verbs_addr laddr,
						     size_t size,
						     doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_put_thread<resource_sharing_mode, nic_handler>(qp, raddr, laddr, size, out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_put_warp<resource_sharing_mode, nic_handler>(qp, raddr, laddr, size, out_ticket);
}

/**
 * @brief Put shared QP function. Redirect to the put switch function. No need for out_ticket.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_put(struct doca_gpu_dev_verbs_qp *qp,
						     struct doca_gpu_dev_verbs_addr raddr,
						     struct doca_gpu_dev_verbs_addr laddr,
						     size_t size)
{
	uint64_t ticket;
	doca_gpu_dev_verbs_put<resource_sharing_mode, nic_handler, exec_scope>(qp, raddr, laddr, size, &ticket);
}

/* **************************************** PUT INLINE **************************************** */
/**
 * @brief Put inline shared QP function. Every thread posts a different RDMA Write Inline in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] value - RDMA Write Inline value to copy
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <typename T,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_thread(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  T value,
							  doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	uint64_t wqe_idx;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

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
	doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp,
													  wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Put inline shared QP function. Every thread in the warp posts a different RDMA Write Inline in a different WQE position.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] value - RDMA Write Inline value to copy
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <typename T,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_warp(struct doca_gpu_dev_verbs_qp *qp,
							struct doca_gpu_dev_verbs_addr raddr,
							T value,
							doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t base_wqe_idx = 0;
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

	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp,
							     wqe_ptr,
							     wqe_idx,
							     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
							     raddr.addr,
							     raddr.key,
							     sizeof(T));
	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<T>(qp, wqe_ptr, value);

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
 * @brief Put inline shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] value - RDMA Write Inline value to copy
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <typename T,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_p(struct doca_gpu_dev_verbs_qp *qp,
						   struct doca_gpu_dev_verbs_addr raddr,
						   T value,
						   doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_p_thread<T, resource_sharing_mode, nic_handler>(qp, raddr, value, out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_p_warp<T, resource_sharing_mode, nic_handler>(qp, raddr, value, out_ticket);
}

/**
 * @brief Put inline shared QP function. Redirect to the put switch function. No need for out_ticket.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 */
template <typename T,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_p(struct doca_gpu_dev_verbs_qp *qp,
						   struct doca_gpu_dev_verbs_addr raddr,
						   T value)
{
	uint64_t ticket;
	doca_gpu_dev_verbs_p<T, resource_sharing_mode, nic_handler, exec_scope>(qp, raddr, value, &ticket);
}

/* **************************************** GET **************************************** */
/**
 * @brief Get shared QP function. Every thread posts a different RDMA Read in a different WQE position.
 * Every thread submits the WQE.
 * On some GPU architectures/systems, and RDMA Read (as well as RDMA Recv) may require to post a WQE Dump
 * before accessing the read/received data to ensure GPU memory consistency.
 * Application may decide function's behaviour about the WQE Dump via the mcst_mode template parameter.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Read remote info
 * @param[in] laddr - RDMA Read local info
 * @param[in] size - RDMA Read size (bytes)
 * @param[in] daddr - Dump local address. Used only if needed.
 * @param[in] out_ticket - WQE index of the RDMA Read or the WQE Dump
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED,
          enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_DISABLED>
__device__ static inline void doca_gpu_dev_verbs_get_thread(struct doca_gpu_dev_verbs_qp *qp,
                                                     struct doca_gpu_dev_verbs_addr raddr,
                                                     struct doca_gpu_dev_verbs_addr laddr,
                                                     size_t size,
								                     struct doca_gpu_dev_verbs_addr daddr,
                                                     doca_gpu_dev_verbs_ticket_t *out_ticket)
{
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t base_wqe_idx;
    uint64_t wqe_idx;
    size_t remaining_size = size;
    size_t size_;
    uint32_t num_chunks = doca_gpu_dev_verbs_div_ceil_aligned_pow2_32bits(
        size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
	bool need_mcst;
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	need_mcst = (bool)__ldg((int *)&qp->need_mcst);
	if (mcst_mode == DOCA_GPUNETIO_VERBS_MCST_ENABLED)
		base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks + 1);
	else
		base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks);

#pragma unroll 1
    for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;

        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
        doca_gpu_dev_verbs_wqe_prepare_read(
            qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
            raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
            laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
        remaining_size -= size_;
    }

	if (mcst_mode == DOCA_GPUNETIO_VERBS_MCST_ENABLED) {
            wqe_idx++;  // Use the reserved slot after all chunks
            wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

            doca_gpu_dev_verbs_wqe_prepare_dump(qp, wqe_ptr, wqe_idx,
                                                DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                                daddr.addr, daddr.key, 1);
    }

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
    doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                              nic_handler>(qp, wqe_idx + 1);

    // Wait for the actual last WQE we submitted (either last GET chunk or DUMP)
    if (blocking_mode == DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED)
        doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), wqe_idx);

    // The actual last WQE we submitted could be either last READ chunk or DUMP
    *out_ticket = wqe_idx;
}

/**
 * @brief Get shared QP function. Every thread in the warp posts a different RDMA Read in a different WQE position.
 * Only thread with lane_idx = 0 submits all the WQE posted by the warp.
 * On some GPU architectures/systems, and RDMA Read (as well as RDMA Recv) may require to post a WQE Dump
 * before accessing the read/received data to ensure GPU memory consistency.
 * Application may decide function's behaviour about the WQE Dump via the mcst_mode template parameter.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Read remote info
 * @param[in] laddr - RDMA Read local info
 * @param[in] size - RDMA Read size (bytes)
 * @param[in] daddr - Dump local address. Used only if needed.
 * @param[in] out_ticket - WQE index of the RDMA Read or the WQE Dump
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED,
          enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_DISABLED>
__device__ static inline void doca_gpu_dev_verbs_get_warp(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  struct doca_gpu_dev_verbs_addr laddr,
							  size_t size,
							  struct doca_gpu_dev_verbs_addr daddr,
							  doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t base_wqe_idx = 0;
	uint64_t wqe_idx;
	uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();

	DOCA_GPUNETIO_VERBS_ASSERT(size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	if (lane_idx == 0) {
		if (mcst_mode == DOCA_GPUNETIO_VERBS_MCST_ENABLED)
			base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(
				qp,
				DOCA_GPUNETIO_VERBS_WARP_SIZE + 1);
		else
			base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(
				qp,
				DOCA_GPUNETIO_VERBS_WARP_SIZE);

		base_wqe_idx_0 = (uint32_t)base_wqe_idx;
		base_wqe_idx_1 = (uint32_t)(base_wqe_idx >> 32);
	}
	__syncwarp();

	base_wqe_idx_0 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_0);
	base_wqe_idx_1 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_1);
	base_wqe_idx = ((uint64_t)base_wqe_idx_1) << 32 | base_wqe_idx_0;

	wqe_idx = base_wqe_idx + lane_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
	doca_gpu_dev_verbs_wqe_prepare_read(
		qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
		raddr.addr, raddr.key,
		laddr.addr, laddr.key, size);
	__syncwarp();

	if (lane_idx == 0) {
		if (mcst_mode == DOCA_GPUNETIO_VERBS_MCST_ENABLED) {
			wqe_idx = base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE;  // Use the reserved slot after all chunks
			wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

			doca_gpu_dev_verbs_wqe_prepare_dump(qp, wqe_ptr, wqe_idx,
												DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
												daddr.addr, daddr.key, 1);
			doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);
			doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp, wqe_idx + 1);
			// Wait for the actual last WQE we submitted (either last GET chunk or DUMP)
			if (blocking_mode == DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED)
				doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), wqe_idx);
		} else {
			doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(
						qp,
						base_wqe_idx,
						base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE - 1);
			doca_gpu_dev_verbs_submit<resource_sharing_mode,
						DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
						nic_handler>(qp, base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE);
			// Wait for the actual last WQE we submitted (either last GET chunk or DUMP)
			if (blocking_mode == DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED)
				doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE - 1);
		}
	}
	__syncwarp();

	// The actual last WQE we submitted could be either last READ chunk or DUMP
	*out_ticket = wqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Get shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Read remote info
 * @param[in] laddr - RDMA Read local info
 * @param[in] size - RDMA Read size (bytes)
 * @param[in] daddr - Dump local address. Used only if needed.
 * @param[in] out_ticket - WQE index of the RDMA Read or the WQE Dump
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED,
          enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_DISABLED>
__device__ static inline void doca_gpu_dev_verbs_get(struct doca_gpu_dev_verbs_qp *qp,
						     struct doca_gpu_dev_verbs_addr raddr,
						     struct doca_gpu_dev_verbs_addr laddr,
						     size_t size,
						     struct doca_gpu_dev_verbs_addr daddr,
						     doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_get_thread<resource_sharing_mode, nic_handler, mcst_mode, blocking_mode>(qp,
											     raddr,
											     laddr,
											     size,
											     daddr,
											     out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_get_warp<resource_sharing_mode, nic_handler, mcst_mode, blocking_mode>(qp,
											   raddr,
											   laddr,
											   size,
											   daddr,
											   out_ticket);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED>
__device__ static inline void doca_priv_gpu_dev_verbs_get_wait(struct doca_gpu_dev_verbs_qp *qp,
                                                            struct doca_gpu_dev_verbs_addr daddr,
                                                            doca_gpu_dev_verbs_ticket_t *ticket) {
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
    }

	doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode>(doca_gpu_dev_verbs_qp_get_cq_sq(qp), *ticket);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED>
__device__ static inline void doca_gpu_dev_verbs_get_wait(struct doca_gpu_dev_verbs_qp *qp,
                                                            struct doca_gpu_dev_verbs_addr daddr,
															doca_gpu_dev_verbs_ticket_t *ticket) {
	doca_priv_gpu_dev_verbs_get_wait<resource_sharing_mode, nic_handler, mcst_mode>(qp, daddr, ticket);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
		  enum doca_gpu_dev_verbs_mcst_mode mcst_mode = DOCA_GPUNETIO_VERBS_MCST_DISABLED>
__device__ static inline void doca_gpu_dev_verbs_get_wait(struct doca_gpu_dev_verbs_qp *qp,
                                                            struct doca_gpu_dev_verbs_addr daddr) {
	uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(&qp->sq_rsvd_index);
	[[unlikely]] if (ticket == 0)
		return;
	--ticket;

	doca_priv_gpu_dev_verbs_get_wait<resource_sharing_mode, nic_handler, mcst_mode>(qp, daddr, &ticket);
}

/* **************************************** PUT SIGNAL **************************************** */
/**
 * @brief Put + signal shared QP function. Every thread posts a different RDMA Write + RDMA Atomic FetchAdd
 * in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_signal_thread(struct doca_gpu_dev_verbs_qp *qp,
								   struct doca_gpu_dev_verbs_addr raddr,
								   struct doca_gpu_dev_verbs_addr laddr,
								   size_t size,
								   struct doca_gpu_dev_verbs_addr sig_raddr,
								   struct doca_gpu_dev_verbs_addr sig_laddr,
								   uint64_t sig_val,
								   doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx;
	uint64_t wqe_idx;
	size_t remaining_size = size;
	size_t size_;
	uint64_t num_chunks =
		doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

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
	doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp,
													  wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Put + signal shared QP function. Every thread in the warp posts a different RDMA Write in a different WQE position.
 * Only thread with lane_idx = 0 posts an additional RDMA Atomic FetchAdd and submits all the WQE posted by the warp.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_put_signal_warp(struct doca_gpu_dev_verbs_qp *qp,
								 struct doca_gpu_dev_verbs_addr raddr,
								 struct doca_gpu_dev_verbs_addr laddr,
								 size_t size,
								 struct doca_gpu_dev_verbs_addr sig_raddr,
								 struct doca_gpu_dev_verbs_addr sig_laddr,
								 uint64_t sig_val,
								 doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t base_wqe_idx = 0;
	uint64_t wqe_idx;
	uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();

	DOCA_GPUNETIO_VERBS_ASSERT(size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	if (lane_idx == 0) {
		base_wqe_idx =
			doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp,
										   DOCA_GPUNETIO_VERBS_WARP_SIZE + 1);
		base_wqe_idx_0 = (uint32_t)base_wqe_idx;
		base_wqe_idx_1 = (uint32_t)(base_wqe_idx >> 32);
	}
	__syncwarp();

	base_wqe_idx_0 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_0);
	base_wqe_idx_1 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_1);
	base_wqe_idx = ((uint64_t)base_wqe_idx_1) << 32 | base_wqe_idx_0;


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
		wqe_idx = base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE;

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
		doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
			qp,
			wqe_idx + 1);
	}
	__syncwarp();

	*out_ticket = wqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Put + signal shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_put_signal(struct doca_gpu_dev_verbs_qp *qp,
							    struct doca_gpu_dev_verbs_addr raddr,
							    struct doca_gpu_dev_verbs_addr laddr,
							    size_t size,
							    struct doca_gpu_dev_verbs_addr sig_raddr,
							    struct doca_gpu_dev_verbs_addr sig_laddr,
							    uint64_t sig_val,
							    doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_put_signal_thread<sig_op, resource_sharing_mode, nic_handler>(qp,
												 raddr,
												 laddr,
												 size,
												 sig_raddr,
												 sig_laddr,
												 sig_val,
												 out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_put_signal_warp<sig_op, resource_sharing_mode, nic_handler>(qp,
											       raddr,
											       laddr,
											       size,
											       sig_raddr,
											       sig_laddr,
											       sig_val,
											       out_ticket);
}

/**
 * @brief Put + signal shared QP function. Redirect to the put switch function. No need for out_ticket.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
__device__ static inline void doca_gpu_dev_verbs_put_signal(struct doca_gpu_dev_verbs_qp *qp,
							    struct doca_gpu_dev_verbs_addr raddr,
							    struct doca_gpu_dev_verbs_addr laddr,
							    size_t size,
							    struct doca_gpu_dev_verbs_addr sig_raddr,
							    struct doca_gpu_dev_verbs_addr sig_laddr,
							    uint64_t sig_val)
{
	uint64_t ticket;
	doca_gpu_dev_verbs_put_signal<sig_op, resource_sharing_mode, nic_handler, exec_scope>(qp,
											      raddr,
											      laddr,
											      size,
											      sig_raddr,
											      sig_laddr,
											      sig_val,
											      &ticket);
}

/**
 * @brief Put inline + signal shared QP function. Every thread posts a different RDMA Write Inline + RDMA Atomic FetchAdd
 * in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <typename T,
	  enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_signal_thread(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  T value,
							  struct doca_gpu_dev_verbs_addr sig_raddr,
							  struct doca_gpu_dev_verbs_addr sig_laddr,
							  uint64_t sig_val,
							  doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint64_t base_wqe_idx;
	uint64_t wqe_idx;

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

	base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 2);
	wqe_idx = base_wqe_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp,
							     wqe_ptr,
							     wqe_idx,
							     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
							     raddr.addr,
							     raddr.key,
							     sizeof(T));
	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<T>(qp, wqe_ptr, value);

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
	doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp,
													  wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Put inline + signal shared QP function. Every thread in the warp posts a different RDMA Write Inline in a different WQE position.
 * Only thread with lane_idx = 0 posts an additional RDMA Atomic FetchAdd and submits all the WQE posted by the warp.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <typename T,
	  enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_signal_warp(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  T value,
							  struct doca_gpu_dev_verbs_addr sig_raddr,
							  struct doca_gpu_dev_verbs_addr sig_laddr,
							  uint64_t sig_val,
							  doca_gpu_dev_verbs_ticket_t *out_ticket)
{
#if __CUDA_ARCH__ >= 800
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;
	uint32_t base_wqe_idx_0 = 0, base_wqe_idx_1 = 0;
	uint64_t base_wqe_idx;
	uint64_t wqe_idx;

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
	if (lane_idx == 0) {
		base_wqe_idx =
			doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp,
										   DOCA_GPUNETIO_VERBS_WARP_SIZE + 1);
		base_wqe_idx_0 = (uint32_t)base_wqe_idx;
		base_wqe_idx_1 = (uint32_t)(base_wqe_idx >> 32);
	}
	__syncwarp();

	base_wqe_idx_0 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_0);
	base_wqe_idx_1 = __reduce_max_sync(DOCA_GPUNETIO_VERBS_WARP_FULL_MASK, base_wqe_idx_1);
	base_wqe_idx = ((uint64_t)base_wqe_idx_1) << 32 | base_wqe_idx_0;

	wqe_idx = base_wqe_idx + lane_idx;
	wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp,
							     wqe_ptr,
							     wqe_idx,
							     DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
							     raddr.addr,
							     raddr.key,
							     sizeof(T));
	doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<T>(qp, wqe_ptr, value);

		__syncwarp();
	if (lane_idx == 0) {
		wqe_idx = base_wqe_idx + DOCA_GPUNETIO_VERBS_WARP_SIZE;

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
		doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
			qp,
			wqe_idx + 1);
	}
	__syncwarp();

	*out_ticket = wqe_idx;
#else
	printf("__CUDA_ARCH__ < 800, WARP mode not enabled\n");
	*out_ticket = 0;
#endif
}

/**
 * @brief Put inline + signal shared QP switch function. Redirect to thread or warp scope.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] raddr - RDMA Write remote info
 * @param[in] laddr - RDMA Write local info
 * @param[in] size - RDMA Write size (bytes)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the last WQE posted
 */
template <typename T,
	  enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_exec_scope exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_signal(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  T value,
							  struct doca_gpu_dev_verbs_addr sig_raddr,
							  struct doca_gpu_dev_verbs_addr sig_laddr,
							  uint64_t sig_val,
							  doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
		doca_gpu_dev_verbs_p_signal_thread<T, sig_op, resource_sharing_mode, nic_handler>(qp,
										   raddr,
										   value,
										   sig_raddr,
										   sig_laddr,
										   sig_val,
										   out_ticket);
	if (exec_scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
		doca_gpu_dev_verbs_p_signal_warp<T, sig_op, resource_sharing_mode, nic_handler>(qp,
										   raddr,
										   value,
										   sig_raddr,
										   sig_laddr,
										   sig_val,
										   out_ticket);

}

template <typename T,
	  enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_p_signal(struct doca_gpu_dev_verbs_qp *qp,
							  struct doca_gpu_dev_verbs_addr raddr,
							  T value,
							  struct doca_gpu_dev_verbs_addr sig_raddr,
							  struct doca_gpu_dev_verbs_addr sig_laddr,
							  uint64_t sig_val)
{
	uint64_t ticket;
	doca_gpu_dev_verbs_p_signal<T, sig_op, resource_sharing_mode, nic_handler>(qp,
										   raddr,
										   value,
										   sig_raddr,
										   sig_laddr,
										   sig_val,
										   &ticket);
}

/* **************************************** SIGNAL **************************************** */
/**
 * @brief Signal shared QP function. Every thread posts a different RDMA Atomic FetchAdd in a different WQE position.
 * Every thread submits the WQE.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 * @param[in] out_ticket - WQE index of the RDMA Write
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_signal(struct doca_gpu_dev_verbs_qp *qp,
							struct doca_gpu_dev_verbs_addr sig_raddr,
							struct doca_gpu_dev_verbs_addr sig_laddr,
							uint64_t sig_val,
							doca_gpu_dev_verbs_ticket_t *out_ticket)
{
	uint64_t wqe_idx;
	struct doca_gpu_dev_verbs_wqe *wqe_ptr;

	DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
	DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

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
	doca_gpu_dev_verbs_submit<resource_sharing_mode, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(qp,
													  wqe_idx + 1);

	*out_ticket = wqe_idx;
}

/**
 * @brief Signal shared QP function. Every thread posts a different RDMA Atomic FetchAdd in a different WQE position.
 * Every thread submits the WQE. No out_ticket.
 *
 * @param[in] qp - Queue Pair (QP)
 * @param[in] sig_raddr - RDMA Atomic remote info
 * @param[in] sig_laddr - RDMA Atomic local info
 * @param[in] sig_val - RDMA Atomic value
 */
template <enum doca_gpu_dev_verbs_signal_op sig_op,
	  enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static inline void doca_gpu_dev_verbs_signal(struct doca_gpu_dev_verbs_qp *qp,
							struct doca_gpu_dev_verbs_addr sig_raddr,
							struct doca_gpu_dev_verbs_addr sig_laddr,
							uint64_t sig_val)
{
	uint64_t ticket;
	doca_gpu_dev_verbs_signal<sig_op, resource_sharing_mode, nic_handler>(qp,
									      sig_raddr,
									      sig_laddr,
									      sig_val,
									      &ticket);
}


#endif /* DOCA_GPUNETIO_DEV_VERBS_ONESIDED_CUH */
