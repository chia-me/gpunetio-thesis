/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * @file doca_gpunetio_dev_buf.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCA_GPUNETIO_DEV_BUF DOCA GPUNetIO Device - Buffer
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_BUF_H
#define DOCA_GPUNETIO_DEVICE_BUF_H

#include <stdint.h>
#include <doca_gpunetio.h>
#include <doca_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structures representing GPU handlers for DOCA buffers.
 */
struct doca_gpu_buf;
struct doca_gpu_buf_arr;

/*********************************************************************************************************************
 * DOCA buffer management
 *********************************************************************************************************************/

/**
 * @brief Retrieve a specific doca_gpu_buf pointer from doca_gpu_buf_arr object.
 * This is a data path function: to make it faster, there is no input data check.
 * Please ensure application is not passing NULL pointers.
 *
 * WARNING: to decrease latency and speed-up data path, this function doesn't test if
 * input arguments are valid. It relies on the application for it.
 *
 * @param [in] buf_arr_gpu
 * doca_gpu_buf_arr object.
 * @param [in] doca_gpu_buf_idx
 * Index of the doca_gpu_buf to retrieve from the receive queue.
 * @param [out] buf
 * doca_gpu_buf pointer.
 *
 * @return
 */
__device__ void doca_gpu_dev_buf_get_buf(const struct doca_gpu_buf_arr *buf_arr_gpu,
					 const uint64_t doca_gpu_buf_idx,
					 struct doca_gpu_buf **buf);

/**
 * @brief Retrieve memory address of the packet stored in the doca_gpu_buf.
 * This is a data path function: to make it faster, there is no input data check.
 * Please ensure application is not passing NULL pointers.
 *
 * WARNING: to decrease latency and speed-up data path, this function doesn't test if
 * input arguments are valid. It relies on the application for it.
 *
 * @param [in] buf
 * doca_gpu_buf
 * @param [out] addr
 * Memory address of the packet in the doca_gpu_buf object
 *
 * @return
 */
__device__ void doca_gpu_dev_buf_get_addr(const struct doca_gpu_buf *buf, uintptr_t *addr);

/**
 * @brief Get mkey with doca_access_flag access for a DOCA GPU buffer.
 * This is a data path function: to make it faster, there is no input data check.
 * Please ensure application is not passing NULL pointers.
 *
 * WARNING: to decrease latency and speed-up data path, this function doesn't test if
 * input arguments are valid. It relies on the application for it.
 *
 * @param [in] buf
 * The DOCA buffer to get mkey for. MUST NOT BE NULL.
 *
 * @param [out] mkey
 * The returned MKey. MUST NOT BE NULL.
 *
 * @note Access of mkey is defined by the mmap where buf was created.
 *
 * @return
 */
__device__ void doca_gpu_dev_buf_get_mkey(const struct doca_gpu_buf *buf, uint32_t *mkey);

/**
 * @brief Create a DOCA GPU buf address with relevant info.
 * This is a data path function: to make it faster, there is no input data check.
 * Please ensure application is not passing NULL pointers.
 *
 * WARNING: to decrease latency and speed-up data path, this function doesn't test if
 * input arguments are valid. It relies on the application for it.
 *
 * @param [in] addr
 * The DOCA GPU buffer address. MUST NOT BE NULL.
 *
 * @param [in] mkey
 * The DOCA GPU buffer mkey. MUST NOT BE NULL.
 *
 * @param [out] buf
 * The doca_gpu_buf filled with info. MUST NOT BE NULL.
 *
 * @return
 */
__device__ void doca_gpu_dev_buf_create(uintptr_t addr, uint32_t mkey, struct doca_gpu_buf **buf);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_GPUNETIO_DEVICE_BUF_H */

/** @} */
