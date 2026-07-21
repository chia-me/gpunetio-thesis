/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * @file doca_gpunetio_dev_dma.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCAGPUNETIO DOCA GPUNetIO engine
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_DMA_H
#define DOCA_GPUNETIO_DEVICE_DMA_H

#include <stdint.h>
#include <doca_gpunetio.h>
#include <doca_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structures representing a DMA GPU queue object.
 */
struct doca_gpu_dma;

/**
 * @brief Enqueue a DMA Memory copy operation in the DMA GPU queue
 *
 * Multiple CUDA threads can invoke this function on the same DMA queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the write operation in sequential position index
 * with respect to other DMA operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] dma - DMA GPU queue handle
 * @param[in] src_buf - src buffer GPU handle
 * @param[in] src_offset - Bytes offset from src buffer initial address
 * @param[in] dst_buf - dst buffer GPU handle
 * @param[in] dst_offset - Bytes offset from dst buffer initial address
 * @param[in] length - Number of bytes to write
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_dma_memcpy(struct doca_gpu_dma *dma,
						struct doca_gpu_buf *src_buf,
						uint64_t src_offset,
						struct doca_gpu_buf *dst_buf,
						uint64_t dst_offset,
						size_t length);

/**
 * @brief Commit and execute the memory copies enqueue in the DMA GPU queue
 *
 * Only 1 CUDA Thread can invoke this function on the same DMA GPU queue at time.
 * Can't be invoked in parallel at the same time by multiple threads.
 *
 * @param[in] dma - DMA GPU queue handle
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_dma_commit(struct doca_gpu_dma *dma);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_GPUNETIO_DEVICE_DMA_H */

/** @} */
