/*
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * @file doca_gpunetio_dev_verbs_cq.cuh
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
#ifndef DOCA_GPUNETIO_DEV_VERBS_CQ_H
#define DOCA_GPUNETIO_DEV_VERBS_CQ_H

#include <doca_gpunetio_dev_verbs_common.cuh>

/**
 * @brief Return device CQ SQ pointer from a device QP
 *
 * @param[in] qp - Dev QP pointer
 *
 * @return Dev CQ pointer
 */
__device__ inline struct doca_gpu_dev_verbs_cq *doca_gpu_dev_verbs_qp_get_cq_sq(struct doca_gpu_dev_verbs_qp *qp)
{
    return &(qp->cq_sq);
}

/**
 * @brief Return device CQ RQ pointer from a device QP
 *
 * @param[in] qp - Dev QP pointer
 *
 * @return Dev CQ pointer
 */
__device__ inline struct doca_gpu_dev_verbs_cq *doca_gpu_dev_verbs_qp_get_cq_rq(struct doca_gpu_dev_verbs_qp *qp)
{
    return &(qp->cq_rq);
}

/**
 * @brief Increment and round up CQE id
 *
 * @param[in] cqe_idx - cqe idx
 * @param[in] increment - cqe idx increment
 *
 * @return cqe incremented idx
 */
__device__ inline uint32_t doca_gpu_dev_verbs_cqe_idx_inc_mask(uint32_t cqe_idx,
                                                                    uint32_t increment) {
    return (cqe_idx + increment) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
}

/**
 * @brief Check if CQE has up to 32B inline data
 *
 * @param[in] cqe64 - CQE memory pointer
 *
 * @return 0 if CQE hasn't inline data, 1 if CQE has inline data.
 */
__device__ inline uint32_t doca_gpu_dev_verbs_cqe_is_inline(struct mlx5_cqe64 *cqe64) {
    return (cqe64->op_own & MLX5_INLINE_SCATTER_32);
}

/**
 * @brief Get CQE number of bytes
 *
 * @param[in] cqe64 - CQE memory pointer
 *
 * @return number of bytes associated to the CQE.
 */
__device__ inline uint32_t doca_gpu_dev_verbs_cqe_get_bytes(struct mlx5_cqe64 *cqe64) {
    return doca_gpu_dev_verbs_bswap32(cqe64->byte_cnt);
}

/**
 * @brief Get CQE inline data
 *
 * @param[in] cqe64 - CQE memory pointer
 *
 * @return pointer to the CQE inline data
 */
__device__ inline uint8_t * doca_gpu_dev_verbs_cqe_get_inl_data(struct mlx5_cqe64 *cqe64) {
    return ((struct doca_gpu_dev_verbs_cqe64_inline *)cqe64)->inl_data;
}

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
/**
 * @brief Print error CQE values
 *
 * @param[in] cqe64 - erroneous cqe
 *
 * @return
 */
__device__ inline void doca_gpu_dev_verbs_cq_print_cqe_err(struct mlx5_cqe64 *cqe64)
{
	struct mlx5_err_cqe_ex *err_cqe = (struct mlx5_err_cqe_ex *)cqe64;

	printf("got completion with err: "
	       "syndrome=%#x, vendor_err_synd=%#x, "
	       "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u wqe_qpn=%x\n",
	       err_cqe->syndrome,
	       err_cqe->vendor_err_synd,
	       err_cqe->hw_err_synd,
	       err_cqe->hw_synd_type,
	       err_cqe->wqe_counter,
	       err_cqe->s_wqe_opcode_qpn);
}
#endif

/**
 * @brief Poll the Completion Queue (CQ) at a specific index returning the pointer to the CQE.
 * This function may or may not wait for the completion to arrive, depending on the blocking_mode
 * template parameter.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @param cqe64 - Pointer passed by reference to get the CQE reference.
 * @return On success, returns 0. If it is a completion with error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ,
      enum doca_gpu_dev_verbs_blocking_mode blocking_mode = DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED>
__device__ static inline int doca_gpu_dev_verbs_poll_cq(struct doca_gpu_dev_verbs_cq *cq,
								uint64_t cons_index, struct mlx5_cqe64 **cqe64)
{
    struct mlx5_cqe64 *cqe =
        (struct mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
    *cqe64 = &cqe[idx];
    uint8_t opown;
    uint8_t opcode;
    uint64_t cqe_ci;

#if __CUDA_ARCH__ >= 900
    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&(*cqe64)->op_own);
    } while (blocking_mode == DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED &&
            ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              ((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num)))));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&(*cqe64)->wqe_counter);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
    } while (blocking_mode == DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED &&
            ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              (((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num)) ||
               (wqe_counter != ((uint32_t)cons_index & 0xffff))))));
#endif

	opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

    if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
		if (opcode == MLX5_CQE_RESP_ERR)
			doca_gpu_dev_verbs_cq_print_cqe_err((*cqe64));
#endif
		if (((opcode == MLX5_CQE_RESP_ERR) * -EIO) == 0) {
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
		    doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
            return 0;
        }

        return (opcode == MLX5_CQE_RESP_ERR) * -EIO;
	} else {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
		if (opcode == MLX5_CQE_REQ_ERR)
			doca_gpu_dev_verbs_cq_print_cqe_err((*cqe64));
#endif
        if (((opcode == MLX5_CQE_REQ_ERR) * -EIO) == 0) {
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
		    doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
            return 0;
        }

        return (opcode == MLX5_CQE_REQ_ERR) * -EIO;
	}
}

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 * It also does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline int doca_priv_gpu_dev_verbs_poll_one_cq_at(struct doca_gpu_dev_verbs_cq *cq,
									 uint64_t cons_index)
{
    uint8_t *cqe = (uint8_t *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
    struct mlx5_cqe64 *cqe64 =
        (struct mlx5_cqe64 *)(cqe + (idx * DOCA_GPUNETIO_VERBS_CQE_SIZE));

    uint64_t cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);

    if (cons_index < cqe_ci) return 0;
    if (cons_index >= cqe_ci + cqe_num) return EBUSY;

    uint8_t opown;
    uint8_t opcode;
    bool observed_completion;

#if __CUDA_ARCH__ >= 900
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);

    observed_completion =
        !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;

    observed_completion =
        !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num)) &&
        (wqe_counter == ((uint32_t)cons_index & 0xffff));
#endif

    if (!observed_completion) return EBUSY;

    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
		if (opcode == MLX5_CQE_RESP_ERR)
			doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
		return (opcode == MLX5_CQE_RESP_ERR) * -EIO;
	} else {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
		if (opcode == MLX5_CQE_REQ_ERR)
			doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
		return (opcode == MLX5_CQE_REQ_ERR) * -EIO;
	}
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function does
 * not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline int doca_gpu_dev_verbs_poll_one_cq_at(struct doca_gpu_dev_verbs_cq *cq,
								    uint64_t cons_index)
{
	int status = doca_priv_gpu_dev_verbs_poll_one_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
	if (status == 0) {
		doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
		doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
	}
	return status;
}

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline int doca_priv_gpu_dev_verbs_poll_cq_at(struct doca_gpu_dev_verbs_cq *cq,
								     uint64_t cons_index)
{
    struct mlx5_cqe64 *cqe =
        (struct mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t idx = cons_index & (cqe_num - 1);
    struct mlx5_cqe64 *cqe64 = &cqe[idx];
    uint8_t opown;
    uint8_t opcode;
    uint64_t cqe_ci;

#if __CUDA_ARCH__ >= 900
    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              ((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num))));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              (((opown & MLX5_CQE_OWNER_MASK) ^ !!(cons_index & cqe_num)) ||
               (wqe_counter != ((uint32_t)cons_index & 0xffff)))));
#endif

	opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

	if (qp_type == DOCA_GPUNETIO_VERBS_QP_RQ) {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
		if (opcode == MLX5_CQE_RESP_ERR)
			doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
		return (opcode == MLX5_CQE_RESP_ERR) * -EIO;
	} else {
#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
		if (opcode == MLX5_CQE_REQ_ERR)
			doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
		return (opcode == MLX5_CQE_REQ_ERR) * -EIO;
	}
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function waits for the completion
 * to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_at() returns 0. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline int doca_gpu_dev_verbs_poll_cq_at(struct doca_gpu_dev_verbs_cq *cq,
								uint64_t cons_index)
{
	int status = doca_priv_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
	if (status == 0) {
		doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
		doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
	}
	return status;
}

/**
 * @brief Poll the Completion Queue (CQ). This function waits for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param count - Number of completions to poll
 * @return On success, doca_gpu_dev_verbs_poll_cq() returns 0. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
		  DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline int doca_gpu_dev_verbs_poll_cq(struct doca_gpu_dev_verbs_cq *cq, uint32_t count)
{
	uint64_t cons_index =
		doca_gpu_dev_verbs_atomic_add<uint64_t, resource_sharing_mode>(&cq->cqe_rsvd, count) + count - 1;
	return doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
}

/**
 * @brief Increment CQ DBREC
 *
 * @param[in] cq - GPU Completion Queue
 * @param[in] cqe_num - CQE num to increment
 *
 * @return new CQE consumer index
 */
template <bool is_overrun>
__device__ inline uint32_t doca_gpu_dev_verbs_cq_update_dbrec(struct doca_gpu_dev_verbs_cq *cq,
								   uint32_t cqe_num)
{
	uint32_t cqe_ci = DOCA_GPUNETIO_VERBS_VOLATILE(cq->cqe_ci);

	cqe_ci = (cqe_ci + cqe_num) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
	if (is_overrun == false) {
		asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
			     :
			     : "l"(cq->dbrec), "r"(doca_gpu_dev_verbs_bswap32(cqe_ci)));
	}

	DOCA_GPUNETIO_VERBS_VOLATILE(cq->cqe_ci) = cqe_ci;

	return cqe_ci;
}


#endif /* DOCA_GPUNETIO_DEV_VERBS_CQ_H */

/** @} */
