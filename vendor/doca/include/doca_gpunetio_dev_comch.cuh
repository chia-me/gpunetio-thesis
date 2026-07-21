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
 * @file doca_gpunetio_dev_comch.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCAGPUNETIO DOCA GPUNetIO engine
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */

#ifndef DOCA_GPU_DEVICE_COMCH_CUH
#define DOCA_GPU_DEVICE_COMCH_CUH

#include <stdint.h>
#include <doca_gpunetio.h>
#include <doca_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque structure representing a GPU ComCh Consumer object */
struct doca_comch_gpu_consumer;

/* Opaque structure representing a GPU ComCh Producer object */
struct doca_comch_gpu_producer;

/*
 * GPU ComCh wait modes
 */
enum doca_gpu_dev_comch_wait_flags {
	DOCA_GPU_COMCH_WAIT_FLAG_NB = 0, /**< Non-Blocking mode: the wait function  checks if any of the operations
					  * have been completed (data has been sent/received) and exit from the
					  * function. If nothing has been sent/received, the function doesn't block
					  * the execution.
					  */
	DOCA_GPU_COMCH_WAIT_FLAG_B = 1,	 /**< Blocking mode: the wait function blocks the execution waiting for a
					  * send/receive operations to be completed.
					  */
};

/**
 * @brief Post a buffer to receive a message from a remote producer
 *
 * @param [in] consumer
 * doca_comch_gpu_consumer handle.
 * @param [in] mkey
 * The MKey of the buffer messages can be received into.
 * @param [in] addr
 * The address of the buffer messages can be received into.
 * @param [in] recv_len
 * The size of the buffer messages can be received into.
 *
 * @return
 * DOCA_SUCCESS - in case of success, buffer has been posted.
 * DOCA_ERROR_AGAIN - if there is not enough space to submit a post recv buffer
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_BAD_STATE - if the consumer is not in the running state
 */
__device__ doca_error_t doca_dev_gpu_comch_consumer_post_recv(struct doca_comch_gpu_consumer *consumer,
							      uint32_t mkey,
							      uintptr_t addr,
							      uint32_t recv_len);

/**
 * @brief Wait for a message to be received from a remote producer
 *
 * @param [in] consumer
 * doca_comch_gpu_consumer handle.
 * @param [out] mkey
 * The MKey of the buffer message has been received into. MUST NOT BE NULL.
 * @param [out] addr
 * The address of the buffer messages has been received into. MUST NOT BE NULL.
 * @param [out] recv_len
 * The size of the buffer messages has been be received into. MUST NOT BE NULL.
 * @param [out] imm_data
 * Pointer to immediate data included with the message.
 * @param [in/out] imm_data_len
 * IN: The maximum size of immediate data that can be received.
 * OUT: The actual size of immediate data received.
 * @param [in] flags
 * Receive mode, doca_gpu_dev_comch_wait_flags enum.
 *
 * @return
 * DOCA_SUCCESS - in case of success, message has been received.
 * DOCA_ERROR_AGAIN - Only in non-blocking mode, success but no message had been received.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_BAD_STATE - if the consumer is not in the running state.
 * * DOCA_ERROR_IO_FAILED - if an error has occurred when receiving message.
 */
__device__ doca_error_t doca_dev_gpu_comch_consumer_recv_wait(struct doca_comch_gpu_consumer *consumer,
							      uint32_t *mkey,
							      uintptr_t *addr,
							      uint32_t *recv_len,
							      uint8_t **imm_data,
							      uint32_t *imm_data_len,
							      const enum doca_gpu_dev_comch_wait_flags flags);


/**
 * @brief Send a message to a remote consumer.
 *
 * @param [in] producer
 * doca_comch_gpu_producer handle.
 * @param [in] mkey
 * The MKey of the message buffer.
 * @param [in] addr
 * The address of the message buffer.
 * @param [in] send_length
 * The size of the message buffer.
 * @param [in] imm_data
 * Pointer to immediate data to be included with the message.
 * @param [in] imm_data_len
 * The size of immediate data.
 * @param [in] consumer_id
 * The ID of the consumer to send the message to.
 * @param [in] user_msg_id
 * Message id that will be returned once the message has been sent
 *
 * @return
 * DOCA_SUCCESS - in case of success, message has been enqueued for sending.
 * DOCA_ERROR_AGAIN - No space available to send message
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_BAD_STATE - if the producer is not in the running state.
 */
__device__ doca_error_t doca_dev_gpu_comch_producer_send(struct doca_comch_gpu_producer *producer,
							 uint32_t mkey,
							 uintptr_t addr,
							 uint32_t send_length,
							 uint8_t *imm_data,
							 uint32_t imm_data_len,
							 uint32_t consumer_id,
							 uint64_t user_msg_id);

/**
 *
 * @brief Poll to check if messages have been sent to a remote consumer.
 *
 * @param [in] producer
 * doca_comch_gpu_producer handle.
 * @param [out] user_msg_id
 * The message id set when the message was submitted
 * @param [in] flags
 * Receive mode, doca_gpu_dev_comch_wait_flags enum.
 *
 * @return
 * DOCA_SUCCESS - in case of success, message has been sent.
 * DOCA_ERROR_AGAIN - Only in non-blocking mode, success but no message had been sent.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_BAD_STATE - if the producer is not in the running state.
 * * DOCA_ERROR_IO_FAILED - if an error has occurred when sending the message.
 */
__device__ doca_error_t doca_dev_gpu_comch_producer_poll(struct doca_comch_gpu_producer *producer,
							 uint64_t *user_msg_id,
							 const enum doca_gpu_dev_comch_wait_flags flags);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOCA_GPUNETIO_DEVICE_COMCH_CUH */

/** @} */
