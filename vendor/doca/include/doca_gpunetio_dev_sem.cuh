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
 * @file doca_gpunetio_dev_sem.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCA_GPUNETIO_DEV_SEM DOCA GPUNetIO Device - Semaphore
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_SEM_H
#define DOCA_GPUNETIO_DEVICE_SEM_H

#include <stdint.h>
#include <doca_gpunetio.h>
#include <doca_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************************************************************************************
 * Semaphore data path
 *********************************************************************************************************************/

/**
 * Set DOCA GPUNetIO semaphore status from GPU.
 *
 * @param [in] semaphore_gpu
 * DOCA GPUNetIO semaphore GPU handler.
 * @param [in] idx
 * DOCA GPUNetIO semaphore item index.
 * @param [in] status
 * DOCA GPUNetIO semaphore status.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
__device__ doca_error_t doca_gpu_dev_semaphore_set_status(struct doca_gpu_semaphore_gpu *semaphore_gpu,
							  uint32_t idx,
							  enum doca_gpu_semaphore_status status);

/**
 * Set DOCA GPUNetIO semaphore status from GPU using an atomic CAS operation.
 *
 * @param [in] semaphore_gpu
 * DOCA GPUNetIO semaphore GPU handler.
 * @param [in] idx
 * DOCA GPUNetIO semaphore item index.
 * @param [in] status_compare
 * DOCA GPUNetIO semaphore status to compare before changing to the new value.
 * @param [in] status_val
 * DOCA GPUNetIO semaphore status new value if status comparison return true.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_PERMITTED - if status_compare doesn't match with actual semaphore status
 */
__device__ doca_error_t doca_gpu_dev_semaphore_set_status_cas(struct doca_gpu_semaphore_gpu *semaphore_gpu,
							      uint32_t idx,
							      enum doca_gpu_semaphore_status status_compare,
							      enum doca_gpu_semaphore_status status_val);

/**
 * Set DOCA GPUNetIO semaphore status, number of packets and first doca gpu buf idx from GPU.
 *
 * @param [in] semaphore_gpu
 * Semaphore GPU handler.
 * @param [in] idx
 * Semaphore item index.
 * @param [in] status
 * Semaphore status.
 * @param [in] num_packets
 * Number of packets.
 * @param [in] doca_buf_idx_start
 * First doca buf index.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
__device__ doca_error_t doca_gpu_dev_semaphore_set_packet_info(const struct doca_gpu_semaphore_gpu *semaphore_gpu,
							       uint32_t idx,
							       enum doca_gpu_semaphore_status status,
							       uint32_t num_packets,
							       uint64_t doca_buf_idx_start);

/**
 * Get DOCA GPUNetIO semaphore status from CPU.
 *
 * @param [in] semaphore_gpu
 * Semaphore GPU handler.
 * @param [in] idx
 * Semaphore item index.
 * @param [out] status
 * Semaphore status.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
__device__ doca_error_t doca_gpu_dev_semaphore_get_status(const struct doca_gpu_semaphore_gpu *semaphore_gpu,
							  uint32_t idx,
							  enum doca_gpu_semaphore_status *status);

/**
 * Get DOCA GPUNetIO semaphore packet info.
 *
 * @param [in] semaphore_gpu
 * Semaphore GPU handler.
 * @param [in] idx
 * Semaphore item index.
 * @param [out] status
 * Semaphore status.
 * @param [out] num_packets
 * Number of packets.
 * @param [out] doca_buf_idx_start
 * First doca buf index.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
__device__ doca_error_t doca_gpu_dev_semaphore_get_packet_info(const struct doca_gpu_semaphore_gpu *semaphore_gpu,
							       uint32_t idx,
							       enum doca_gpu_semaphore_status *status,
							       uint32_t *num_packets,
							       uint64_t *doca_buf_idx_start);

/**
 * Get DOCA GPUNetIO semaphore packet info if the semaphore status matches
 * the status provided as input parameter.
 *
 * @param [in] semaphore_gpu
 * Semaphore GPU handler.
 * @param [in] idx
 * Semaphore item index.
 * @param [in] status
 * Semaphore status to check.
 * @param [out] num_packets
 * Number of packets.
 * @param [out] doca_buf_idx_start
 * First doca buf index.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_FOUND - if semaphore status doesn't match input status
 */
__device__ doca_error_t doca_gpu_dev_semaphore_get_packet_info_status(const struct doca_gpu_semaphore_gpu *semaphore_gpu,
								      uint32_t idx,
								      enum doca_gpu_semaphore_status status,
								      uint32_t *num_packets,
								      uint64_t *doca_buf_idx_start);

/**
 * Get pointer to DOCA GPUNetIO semaphore item associated custom info.
 *
 * @param [in] semaphore_gpu
 * DOCA GPUNetIO semaphore GPU handler.
 * @param [in] idx
 * DOCA GPUNetIO semaphore item index.
 * @param [out] custom_info
 * DOCA GPUNetIO semaphore custom info pointer.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
__device__ doca_error_t doca_gpu_dev_semaphore_get_custom_info_addr(struct doca_gpu_semaphore_gpu *semaphore_gpu,
								    uint32_t idx,
								    void **custom_info);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_GPUNETIO_DEVICE_SEM_H */

/** @} */
