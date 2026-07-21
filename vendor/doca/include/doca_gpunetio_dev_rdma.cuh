/*
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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
 * @file doca_gpunetio_dev_rdma.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCA_GPUNETIO_DEV_RDMA DOCA GPUNetIO Device - RDMA
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_RDMA_H
#define DOCA_GPUNETIO_DEVICE_RDMA_H

#include <stdint.h>
#include <doca_gpunetio.h>
#include <doca_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque structures representing a RDMA GPU queue object.
 */
struct doca_gpu_dev_rdma;

/**
 * Opaque structures representing a RDMA GPU receive queue object.
 */
struct doca_gpu_dev_rdma_r;

/**
 * Opaque structures representing a RDMA GPU recv operation.
 */
struct doca_gpu_rdma_recv_h;

/**
 * RDMA Write operations modes
 */
enum doca_gpu_dev_rdma_write_flags {
	DOCA_GPU_RDMA_WRITE_FLAG_NONE = 0,
	DOCA_GPU_RDMA_WRITE_FLAG_IMM = 1,
};

/**
 * RDMA Send operations modes
 */
enum doca_gpu_dev_rdma_send_flags {
	DOCA_GPU_RDMA_SEND_FLAG_NONE = 0,
	DOCA_GPU_RDMA_SEND_FLAG_IMM = 1,
};

/**
 * RDMA Recv wait modes
 */
enum doca_gpu_dev_rdma_recv_wait_flags {
	DOCA_GPU_RDMA_RECV_WAIT_FLAG_NB = 0, /**< Non-Blocking mode: the wait receive function
					      * doca_gpu_dev_rdma_recv_wait_all checks if any of the receive operations
					      * completed (data has been received) and exit from the function. If
					      * nothing has been received, the function doesn't block the execution.
					      */
	DOCA_GPU_RDMA_RECV_WAIT_FLAG_B = 1,  /**< Blocking mode: the wait receive function
					      * doca_gpu_dev_rdma_recv_wait_all  blocks the execution waiting for all the
					      * receive operations to be completed.
					      */
};

/**
 * @brief Enqueue an RDMA Write operation in the RDMA GPU queue - Weak mode
 *
 * Multiple CUDA threads can invoke this function on the same RDMA queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the write operation in sequential position index
 * with respect to other RDMA operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] remote_buf - remote buffer GPU handle
 * @param[in] remote_offset - Bytes offset from remote buffer initial address
 * @param[in] local_buf - local buffer GPU handle
 * @param[in] local_offset - Bytes offset from local buffer initial address
 * @param[in] length - Number of bytes to write
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - write mode from enum doca_gpu_dev_rdma_write_flags
 * @param[in] position - Position (index) of the write in the RDMA queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_write_weak(struct doca_gpu_dev_rdma *rdma,
						     uint32_t connection_index,
						     struct doca_gpu_buf *remote_buf,
						     uint64_t remote_offset,
						     struct doca_gpu_buf *local_buf,
						     uint64_t local_offset,
						     size_t length,
						     uint32_t imm,
						     const enum doca_gpu_dev_rdma_write_flags flags,
						     uint32_t position);

/**
 * @brief Enqueue an Inline RDMA Write operation in the RDMA GPU queue - Weak mode
 *
 * Multiple CUDA threads can invoke this function on the same RDMA queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the write operation in sequential position index
 * with respect to other RDMA operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] remote_buf - remote buffer GPU handle
 * @param[in] remote_offset - Bytes offset from remote buffer initial address
 * @param[in] inl_data - inline data buffer
 * @param[in] inl_lenght - inline data bytes
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - write mode from enum doca_gpu_dev_rdma_write_flags
 * @param[in] position - Position (index) of the write in the RDMA queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_write_inline_weak(struct doca_gpu_dev_rdma *rdma,
							    uint32_t connection_index,
							    struct doca_gpu_buf *remote_buf,
							    uint64_t remote_offset,
							    uint8_t *inl_data,
							    uint32_t inl_lenght,
							    uint32_t imm,
							    const enum doca_gpu_dev_rdma_write_flags flags,
							    uint32_t position);

/**
 * @brief Enqueue an RDMA Write operation in the RDMA GPU queue - Strong mode
 *
 * RDMA operation position is derived internally by the function.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] remote_buf - remote buffer GPU handle
 * @param[in] remote_offset - Bytes offset from remote buffer initial address
 * @param[in] local_buf - local buffer GPU handle
 * @param[in] local_offset - Bytes offset from local buffer initial address
 * @param[in] length - Number of bytes to write
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - write mode from enum doca_gpu_dev_rdma_write_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_write_strong(struct doca_gpu_dev_rdma *rdma,
						       uint32_t connection_index,
						       struct doca_gpu_buf *remote_buf,
						       uint64_t remote_offset,
						       struct doca_gpu_buf *local_buf,
						       uint64_t local_offset,
						       size_t length,
						       uint32_t imm,
						       const enum doca_gpu_dev_rdma_write_flags flags);

/**
 * @brief Enqueue an Inline RDMA Write operation in the RDMA GPU queue - Strong mode
 * Max allowed size for inline data is 28 bytes.
 * RDMA operation position is derived internally by the function.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] remote_buf - remote buffer GPU handle
 * @param[in] remote_offset - Bytes offset from remote buffer initial address
 * @param[in] inl_data - inline data buffer
 * @param[in] inl_lenght - inline data bytes
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - write mode from enum doca_gpu_dev_rdma_write_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_write_inline_strong(struct doca_gpu_dev_rdma *rdma,
							      uint32_t connection_index,
							      struct doca_gpu_buf *remote_buf,
							      uint64_t remote_offset,
							      uint8_t *inl_data,
							      uint32_t inl_lenght,
							      uint32_t imm,
							      const enum doca_gpu_dev_rdma_write_flags flags);

/**
 * @brief Enqueue an RDMA Read operation in the RDMA GPU queue - Weak mode
 *
 * Multiple CUDA threads can invoke this function on the same RDMA queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the read operation in sequential position index
 * with respect to other RDMA operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] remote_buf - remote buffer GPU handle
 * @param[in] remote_offset - Bytes offset from remote buffer initial address
 * @param[in] local_buf - local buffer GPU handle
 * @param[in] local_offset - Bytes offset from local buffer initial address
 * @param[in] length - Number of bytes to read
 * @param[in] flags_bitmask - reserved for future use. Must be 0.
 * @param [in] position - Position (index) of the read in the RDMA queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_read_weak(struct doca_gpu_dev_rdma *rdma,
						    uint32_t connection_index,
						    struct doca_gpu_buf *remote_buf,
						    uint64_t remote_offset,
						    struct doca_gpu_buf *local_buf,
						    uint64_t local_offset,
						    size_t length,
						    const uint32_t flags_bitmask,
						    uint32_t position);

/**
 * @brief Enqueue an RDMA Read operation in the RDMA GPU queue - Strong mode
 *
 * RDMA operation position is derived internally by the function.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] remote_buf - remote buffer GPU handle
 * @param[in] remote_offset - Bytes offset from remote buffer initial address
 * @param[in] local_buf - local buffer GPU handle
 * @param[in] local_offset - Bytes offset from local buffer initial address
 * @param[in] length - Number of bytes to read
 * @param[in] flags_bitmask - reserved for future use. Must be 0.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_read_strong(struct doca_gpu_dev_rdma *rdma,
						      uint32_t connection_index,
						      struct doca_gpu_buf *remote_buf,
						      uint64_t remote_offset,
						      struct doca_gpu_buf *local_buf,
						      uint64_t local_offset,
						      size_t length,
						      const uint32_t flags_bitmask);

/**
 * @brief Enqueue an RDMA Send operation in the RDMA GPU queue - Weak mode
 *
 * Multiple CUDA threads can invoke this function on the same RDMA queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the send operation in sequential position index
 * with respect to other RDMA operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] send_buf - source buffer GPU handle
 * @param[in] send_offset - Bytes offset from source buffer initial address
 * @param[in] src_length - Number of bytes to read
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - reserved for future use. Must be 0.
 * @param [in] position - Position (index) of the send in the RDMA queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_send_weak(struct doca_gpu_dev_rdma *rdma,
						    uint32_t connection_index,
						    struct doca_gpu_buf *send_buf,
						    uint64_t send_offset,
						    size_t src_length,
						    uint32_t imm,
						    const enum doca_gpu_dev_rdma_send_flags flags,
						    uint32_t position);

/**
 * @brief Enqueue an Inline RDMA Send operation in the RDMA GPU queue - Weak mode
 * Max allowed size for inline data is 44 bytes.
 * Multiple CUDA threads can invoke this function on the same RDMA queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the send operation in sequential position index
 * with respect to other RDMA operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] inl_data - inline data
 * @param[in] inl_lenght - inline bytes
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - reserved for future use. Must be 0.
 * @param [in] position - Position (index) of the send in the RDMA queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_send_inline_weak(struct doca_gpu_dev_rdma *rdma,
							   uint32_t connection_index,
							   uint8_t *inl_data,
							   uint32_t inl_lenght,
							   uint32_t imm,
							   const enum doca_gpu_dev_rdma_send_flags flags,
							   uint32_t position);

/**
 * @brief Enqueue an RDMA Send operation in the RDMA GPU queue - Strong mode
 *
 * RDMA operation position is derived internally by the function.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] send_buf - source buffer GPU handle
 * @param[in] send_offset - Bytes offset from source buffer initial address
 * @param[in] src_length - Number of bytes to send
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - send mode from enum doca_gpu_dev_rdma_send_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_send_strong(struct doca_gpu_dev_rdma *rdma,
						      uint32_t connection_index,
						      struct doca_gpu_buf *send_buf,
						      uint64_t send_offset,
						      size_t src_length,
						      uint32_t imm,
						      const enum doca_gpu_dev_rdma_send_flags flags);

/**
 * @brief Enqueue an Inline RDMA Send operation in the RDMA GPU queue - Strong mode
 * Max allowed size for inline data is 44 bytes.
 * RDMA operation position is derived internally by the function.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] inl_data - inline data
 * @param[in] inl_lenght - inline bytes
 * @param[in] imm - Immediate 32bit value, big endian format
 * @param[in] flags - send mode from enum doca_gpu_dev_rdma_send_flags
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_send_inline_strong(struct doca_gpu_dev_rdma *rdma,
							     uint32_t connection_index,
							     uint8_t *inl_data,
							     uint32_t inl_lenght,
							     uint32_t imm,
							     const enum doca_gpu_dev_rdma_send_flags flags);

/**
 * @brief Create a commit point in the RDMA Recv queue - Weak mode
 *
 * Only one CUDA thread can invoke this function after a sequence of doca_gpu_dev_rdma_*()
 * invoked by multiple threads.
 *
 * @param[in] rdma - RDMA GPU handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param[in] num_ops - Number of RDMA operations enqueued since last commit.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 *
 */
__device__ doca_error_t doca_gpu_dev_rdma_commit_weak(struct doca_gpu_dev_rdma *rdma,
						      uint32_t connection_index,
						      uint32_t num_ops);

/**
 * @brief Create a commit point in the RDMA Recv queue - Strong mode
 *
 * Only one CUDA thread can invoke this function after a sequence of doca_gpu_dev_rdma_*()
 * invoked by multiple threads.
 *
 * @param[in] rdma - RDMA GPU handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_commit_strong(struct doca_gpu_dev_rdma *rdma, uint32_t connection_index);

/**
 * @brief Wait for the completion of all the previous RDMA Write, Read or Send committed
 * on the RDMA queue. The output parameter num_commits returns the number of commits operation completed.
 * This function is not mandatory if the application doesn't
 * need to wait for the completion of posted RDMA operations before moving forward
 * with other RDMA operations.
 *
 * Only one CUDA thread can invoke this function after doca_gpu_dev_rdma_commit().
 *
 * @param[in] rdma - RDMA GPU handle
 * @param[out] num_commits - Number of commit operations completed.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_wait_all(struct doca_gpu_dev_rdma *rdma, uint32_t *num_commits);

/**
 * @brief Get latest occupied position in the RDMA queue and max position index
 * that can be used in the queue.
 *
 * Highly recommended to use in case of weak pattern.
 * Please ensure to query this function before starting to enqueue RDMA operations in weak mode.
 * The library updated the curr_position value after every doca_gpu_dev_rdma_commit_*()
 * The position index should be always masked with mask_max_position value.
 *
 * @param[in] rdma - RDMA GPU queue handle
 * @param[in] connection_index - the RDMA connection index. For single RDMA connection, should be 0.
 * For multiple RDMA connections should be queried using doca_rdma_connection_get_id().
 * @param [in] curr_position - Next available packet position in the queue.
 * @param [in] mask_max_position - Mask of the max index position usable in this queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_get_info(struct doca_gpu_dev_rdma *rdma,
						   uint32_t connection_index,
						   uint32_t *curr_position,
						   uint32_t *mask_max_position);

/**
 * @brief Get an RDMA Recv queue handler from the RDMA GPU handler.
 *
 * @param[in] rdma - RDMA GPU queue handler
 * @param[out] rdma_r - RDMA Recv GPU queue handler
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_get_recv(struct doca_gpu_dev_rdma *rdma, struct doca_gpu_dev_rdma_r **rdma_r);

/**
 * @brief Post a RDMA Recv operation in the RDMA GPU queue - Weak mode
 *
 * Multiple CUDA threads can invoke this function on the same RDMA Recv queue
 * to enqueue operations simultaneously in different positions.
 * It's developer responsibility to enqueue the recv operation in sequential position index
 * with respect to other RDMA Recv operations enqueued by other CUDA threads in the queue.
 *
 * @param[in] rdma_r - RDMA Recv GPU queue handle
 * @param[in] recv_buf - receive buffer GPU handle
 * @param[in] recv_length - Number of bytes to receive
 * @param[in] recv_offset - Number of bytes offset where to receive the packet
 * @param[in] flags_bitmask - reserved for future use. Must be 0.
 * @param [in] position - Position (index) of the recv in the RDMA queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_recv_weak(struct doca_gpu_dev_rdma_r *rdma_r,
						    struct doca_gpu_buf *recv_buf,
						    size_t recv_length,
						    uint64_t recv_offset,
						    const uint32_t flags_bitmask,
						    uint32_t position);

/**
 * @brief Post a RDMA Recv operation in the RDMA GPU queue - Strong mode
 *
 * @param[in] rdma_r - RDMA Recv GPU queue handle
 * @param[in] recv_buf - receive buffer GPU handle
 * @param[in] recv_length - Number of bytes to receive
 * @param[in] recv_offset - Number of bytes offset where to receive the packet
 * @param[in] flags_bitmask - reserved for future use. Must be 0.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_recv_strong(struct doca_gpu_dev_rdma_r *rdma_r,
						      struct doca_gpu_buf *recv_buf,
						      size_t recv_length,
						      uint64_t recv_offset,
						      const uint32_t flags_bitmask);

/**
 * @brief Commit the RDMA Recv queue executing the operations enqueued in it - Weak mode
 *
 * Only one CUDA thread can invoke this function after a sequence of doca_gpu_dev_rdma_*()
 * invoked by multiple threads.
 *
 * @param[in] rdma_r - RDMA Recv GPU queue handle
 * @param[in] num_ops - Number of RDMA Recv operations enqueued since last commit.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 *
 */
__device__ doca_error_t doca_gpu_dev_rdma_recv_commit_weak(struct doca_gpu_dev_rdma_r *rdma_r, uint32_t num_ops);

/**
 * @brief Create a commit point in the RDMA Recv queue - Strong mode
 *
 * Must be invoked after enqueuing multiple items in the RDMA queue with doca_gpu_dev_rdma_recv() functions.
 * After this function the doca_gpu_dev_rdma_recv_wait_all() should be invoked.
 * It's discouraged but not prohibited to invoke more doca_gpu_dev_rdma_recv() and then another
 * doca_gpu_dev_rdma_recv_commit() right after.
 * Only one CUDA thread can invoke this function after a sequence of doca_gpu_dev_rdma_recv()
 * invoked by multiple threads.
 *
 * @param[in] rdma_r - RDMA Recv GPU queue handle
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_recv_commit_strong(struct doca_gpu_dev_rdma_r *rdma_r);

/**
 * @brief Wait for the completion of all the RDMA Recv operation in the RDMA GPU queue.
 *
 * This function waits for the completion of the RDMA Recv operations posted on the queue and returns,
 * in the output parameter, the number of completions detected.
 *
 * If applications is interested in getting the immediate value received from remote send or writes,
 * it can provide as input the imm_val buffer where the function can put immediate values.
 * WARNING: It's application responsibility to properly allocate and dimension this buffer (max size should be the
 * number) of receive operation posted before last commit. If imm_val is nullptr, it will be ignored.
 *
 * Can be used after doca_gpu_dev_rdma_recv_commit().
 * Only one CUDA thread can invoke this function.
 *
 * @param[in] rdma_r - RDMA Recv GPU queue handle
 * @param[in] flags - blocking (DOCA_GPUNETIO_RDMA_RECV_WAIT_BLOCKING) or non-blocking wait (0)
 * @param[out] num_ops - Number of completed receive operations
 * @param[out] imm_val - Store, for each operation, the immediate value received (if any). It can be nullptr.
 * @param[out] connection_index - Store, for each operation, the connection index associated to the receive op. It can
 * be nullptr.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_recv_wait_all(struct doca_gpu_dev_rdma_r *rdma_r,
							const enum doca_gpu_dev_rdma_recv_wait_flags flags,
							uint32_t *num_ops,
							uint32_t *imm_val,
							uint32_t *connection_index);

/**
 * @brief Get latest occupied position in the RDMA Recv queue and max position index
 * that can be used in the queue. Highly recommended to use in case of weak send pattern.
 *
 * Please ensure to query this function before starting to enqueue RDMA Recv operations in weak mode.
 * The library updated the curr_position value after every doca_gpu_dev_rdma_recv_commit_*()
 * The position index should be always masked with mask_max_position value.
 *
 * @param[in] rdma_r - RDMA Recv GPU queue handle
 * @param [in] curr_position - Next available packet position in the queue.
 * @param [in] mask_max_position - Mask of the max index position usable in this queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 */
__device__ doca_error_t doca_gpu_dev_rdma_recv_get_info(struct doca_gpu_dev_rdma_r *rdma_r,
							uint32_t *curr_position,
							uint32_t *mask_max_position);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_GPUNETIO_DEVICE_RDMA_H */

/** @} */
