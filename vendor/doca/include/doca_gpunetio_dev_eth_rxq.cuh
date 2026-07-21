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
 * @file doca_gpunetio_dev_eth_rxq.cuh
 * @page DOCA_GPUNetIO CUDA Device functions
 * @defgroup DOCA_GPUNETIO_DEV_ETH_RXQ DOCA GPUNetIO Device - Ethernet RXQ
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEVICE_ETH_RXQ_H
#define DOCA_GPUNETIO_DEVICE_ETH_RXQ_H

#include <doca_gpunetio_dev_eth_common.cuh>

/**
 * Default value used by recv function for first packet if not initialized.
 */
#define DOCA_GPUNETIO_ETH_RX_NO_PKT 0xFFFFFFFFFFFFFFFF

/**
 * Enable extended attributes for recv functions
 */
enum doca_gpu_dev_eth_rxq_attr_type {
	DOCA_GPUNETIO_ETH_RX_ATTR_NONE = 0, /**< No Rxq extra attributes */
	DOCA_GPUNETIO_ETH_RX_ATTR_TS,	    /**< Fill only CQE timestamp */
	DOCA_GPUNETIO_ETH_RX_ATTR_ALL	    /**< Fill all Rxq attributes */
};

/**
 * Received packet attributes
 */
struct doca_gpu_dev_eth_rxq_attr {
	uint64_t timestamp_ns; /**< Receive timestamp in CQE */
	uint32_t bytes;	       /**< Number of received bytes */
	uint32_t padding;      /**< Reserved for future use  */
};

/**
 * Describes GPUNetIO dev WQE dump for Rxq.
 */
struct doca_gpu_dev_eth_rxq_dump_wqe {
	struct doca_gpu_dev_eth_wqe_ctrl_seg cseg; /**< WQE control segment */
	struct mlx5_wqe_data_seg dseg;		   /**< WQE data segment */
	struct mlx5_wqe_data_seg pad0;		   /**< padding to 64B */
	struct mlx5_wqe_data_seg pad1;		   /**< padding to 64B */
};

/**
 * @brief Enable memory consistency (mcst) mechanism through dump wqe (local read) to the GPU memory after a receive
 * operation. This function can be used only if the dump QP has been created and associated to the Rx QP. The mcst is
 * required in case of pre-Hopper GPUs.
 *
 * @param mcst_qp
 * GPU memory consistency Queue Pair (QP)
 */
template <enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO>
__device__ inline doca_error_t doca_gpu_dev_eth_rxq_mcst(struct doca_gpu_eth_mcst_qp *mcst_qp)
{
	volatile struct doca_gpu_dev_eth_rxq_dump_wqe *wqe;
	struct mlx5_cqe64 *cqe = (struct mlx5_cqe64 *)__ldg((uintptr_t *)&mcst_qp->cqe_addr);
	uint8_t opown, opcode;
	uint64_t wqe_pi = DOCA_GPUNETIO_ETH_VOLATILE(mcst_qp->wqe_pi);
	uint64_t cqe_ci = DOCA_GPUNETIO_ETH_VOLATILE(mcst_qp->cqe_ci);
	const uint32_t cqe_num = __ldg(&mcst_qp->cqe_num);
	const uint32_t cqe_mask = __ldg(&mcst_qp->cqe_mask);

	// Update CSEG
	wqe = &(((volatile struct doca_gpu_dev_eth_rxq_dump_wqe *)mcst_qp->wqe_addr)[wqe_pi & mcst_qp->wqe_mask]);
	wqe->cseg.opmod_idx_opcode = doca_gpu_dev_eth_bswap32((wqe_pi << 8) | DOCA_GPUNETIO_ETH_MLX5_OPCODE_DUMP);

	// DBREC
	doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
	__be32 dbrec_val = doca_gpu_dev_eth_prepare_dbr(wqe_pi);
	__be32 *dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&mcst_qp->wqe_db_rec);
	cuda::atomic_ref<__be32, cuda::thread_scope_system> dbrec_qp_ptr_aref(*dbrec_ptr);
	dbrec_qp_ptr_aref.store(dbrec_val, cuda::std::memory_order_relaxed);

	// DB
	__be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&mcst_qp->wqe_db);
	__be64 db_val = (__be64) * (uint64_t *)&(wqe->cseg);

	if (nic_handler == DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO) {
		const enum doca_gpu_dev_eth_nic_handler qp_nic_handler =
			(enum doca_gpu_dev_eth_nic_handler)__ldg((int *)&mcst_qp->nic_handler);
		if (qp_nic_handler == DOCA_GPUNETIO_ETH_NIC_HANDLER_GPU_SM_DB)
#ifdef DOCA_GPUNETIO_ETH_HAS_STORE_RELAXED_MMIO
		{
			doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
			doca_gpu_dev_eth_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_val);
		}
#else
		{
			cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
			doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
			db_ptr_aref.store(db_val, cuda::std::memory_order_relaxed);
		}
#endif
		else {
			cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
			doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
			db_ptr_aref.store(db_val, cuda::std::memory_order_relaxed);
		}
	} else if (nic_handler == DOCA_GPUNETIO_ETH_NIC_HANDLER_GPU_SM_DB)
#ifdef DOCA_GPUNETIO_ETH_HAS_STORE_RELAXED_MMIO
	{
		doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
		doca_gpu_dev_eth_store_relaxed_mmio((uint64_t *)db_ptr, (uint64_t)db_val);
	}
#else
	{
		cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
		doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
		db_ptr_aref.store(db_val, cuda::std::memory_order_relaxed);
	}
#endif
	else {
		cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
		doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
		db_ptr_aref.store(db_val, cuda::std::memory_order_relaxed);
	}

	// CQE polling
	volatile struct mlx5_cqe64 *cqe64 = (volatile struct mlx5_cqe64 *)&cqe[cqe_ci & cqe_mask];
	do {
		opown = doca_gpu_dev_eth_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
		opcode = opown >> DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT;
		if ((opcode != MLX5_CQE_INVALID) && !((opown & MLX5_CQE_OWNER_MASK) ^ !!(cqe_ci & cqe_num))) {
			if (opcode == (uint8_t)MLX5_CQE_REQ_ERR) {
				doca_gpu_dev_eth_print_cqe_err(cqe64);
				return DOCA_ERROR_UNEXPECTED;
			}
			break;
		}
	} while (1);

	// CQE DBREC update
	doca_gpu_dev_eth_fence_release<DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>();
	dbrec_val = doca_gpu_dev_eth_prepare_cq_dbr(cqe_ci);
	dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&mcst_qp->cq_db_rec);
	cuda::atomic_ref<__be32, cuda::thread_scope_system> dbrec_cq_ptr_aref(*dbrec_ptr);
	dbrec_cq_ptr_aref.store(dbrec_val, cuda::std::memory_order_relaxed);

	// WQE and CQE index update
	DOCA_GPUNETIO_ETH_VOLATILE(mcst_qp->wqe_pi) = wqe_pi + 1;
	DOCA_GPUNETIO_ETH_VOLATILE(mcst_qp->cqe_ci) = cqe_ci + 1;

	return DOCA_SUCCESS;
}

/**
 * @brief Submit the NIC DBREC (Doorbell Record) for RQ
 *
 * @param [in] rxq - GPU Rx Queue Pair (QP)
 * @param [in] prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_rxq_submit_dbr(struct doca_gpu_eth_rxq *rxq, uint64_t prod_index)
{
	__be32 dbrec_val = doca_gpu_dev_eth_prepare_dbr(prod_index);
	DOCA_GPUNETIO_ETH_VOLATILE(*((uint32_t *)(rxq->wqe_db_rec))) = dbrec_val;
	DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) = prod_index;
}

/**
 * @brief Submit the NIC DBREC (Doorbell Record) for CQ
 *
 * @param [in] rxq - GPU Rx Queue Pair (QP)
 * @param [in] prod_index - Producer index
 */
template <enum doca_gpu_dev_eth_resource_sharing_mode resource_sharing_mode = DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU,
	  enum doca_gpu_dev_eth_sync_scope sync_scope = DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU>
__device__ static inline void doca_gpu_dev_eth_rxq_submit_cq_dbr(struct doca_gpu_eth_rxq *rxq, uint64_t prod_index)
{
	__be32 dbrec_val = doca_gpu_dev_eth_prepare_cq_dbr(prod_index);
	DOCA_GPUNETIO_ETH_VOLATILE(*((uint32_t *)(rxq->cq_db_rec))) = dbrec_val;
	DOCA_GPUNETIO_ETH_VOLATILE(rxq->cqe_ci) = prod_index;
}

/*********************************************************************************************************************
 * Ethernet Receive
 *********************************************************************************************************************/

/**
 * @brief Receive packets through the GPU handler of an Ethernet rxq object.
 * This function must be invoked per-thread.
 * It's developer responsibility to ensure each thread invoking this function operates on a different Ethernet receive
 * object. Function will return upon receiving the indicated maximum number of packets or waiting the indicated number
 * of nanoseconds. If timeout is 0, it's ignored and only the maximum number of packets indicates the exit condition.
 *
 * @param [in] rxq
 * GPU handler for Ethernet receive queue.
 * @param [in] max_rx_pkts
 * Max number of packets to receive. If 0, no limit to the number of packets.
 * @param [in] timeout_ns
 * Max number of nanoseconds to wait before exit from the receive. If 0, no limit to the time spent in function.
 * @param [out] out_first_pkt_idx
 * Index of the first received packet.
 * @param [out] out_pkt_num
 * Total number of received packets.
 * @param [out] out_attr
 * Per-packet attributes. Filled only if rx_attr != DOCA_GPUNETIO_ETH_RX_ATTR_NONE.
 * Caller must ensure the size is great enough to hold stats of all received packets.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 *
 */
template <enum doca_gpu_dev_eth_mcst_mode mcst_mode = DOCA_GPUNETIO_ETH_MCST_DISABLED,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_eth_rxq_attr_type rx_attr = DOCA_GPUNETIO_ETH_RX_ATTR_NONE>
__device__ inline doca_error_t doca_gpu_dev_eth_rxq_recv_thread(struct doca_gpu_eth_rxq *rxq,
								uint32_t max_rx_pkts,
								uint64_t timeout_ns,
								uint64_t *out_first_pkt_idx,
								uint32_t *out_pkt_num,
								struct doca_gpu_dev_eth_rxq_attr *out_attr)
{
	const uint64_t cqe_ci = DOCA_GPUNETIO_ETH_VOLATILE(rxq->cqe_ci);
	const uint32_t cqe_num = __ldg(&rxq->cqe_num);
	const uint32_t cqe_mask = __ldg(&rxq->cqe_mask);
	const struct mlx5_cqe64 *cqe = (struct mlx5_cqe64 *)__ldg((uintptr_t *)&rxq->cqe_addr);

	doca_error_t status = DOCA_SUCCESS;
	volatile struct mlx5_cqe64 *cqe64;
	uint8_t opown, opcode;
	uint16_t wqe_id_max = 0;
	uint32_t packet_idx = 0;
	unsigned long long rx_start = 0, rx_now = 0;

	DOCA_GPUNETIO_ETH_VOLATILE(*out_pkt_num) = 0;
	if (timeout_ns > 0)
		rx_start = doca_gpu_dev_eth_query_globaltimer();

	while (1) {
		cqe64 = (volatile struct mlx5_cqe64 *)&cqe[(cqe_ci + packet_idx) & cqe_mask];
		opown = doca_gpu_dev_eth_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
		opcode = opown >> DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT;
		if ((opcode != MLX5_CQE_INVALID) &&
		    !((opown & MLX5_CQE_OWNER_MASK) ^ !!((cqe_ci + packet_idx) & cqe_num))) {
			if (opcode == (uint8_t)MLX5_CQE_RESP_ERR || opcode == (uint8_t)MLX5_CQE_REQ_ERR) {
				doca_gpu_dev_eth_print_cqe_err(cqe64);
				status = DOCA_ERROR_DRIVER;
				break;
			}

			// Keep track of the first received packet
			if (packet_idx == 0) {
				if (rxq->striding_rq)
					*out_first_pkt_idx = ((wqe_id_max & rxq->wqe_mask) * rxq->wqe_strides_num) +
							     DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_counter);
				else
					*out_first_pkt_idx = DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_counter) &
							     rxq->wqe_mask;
			}

			if (rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_ALL) {
				if (rxq->striding_rq)
					out_attr[packet_idx].bytes = doca_gpu_dev_eth_bswap32(cqe64->byte_cnt) & 0xFFFF;
				else
					out_attr[packet_idx].bytes = doca_gpu_dev_eth_bswap32(cqe64->byte_cnt);
			}

			if (rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_ALL || rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_TS) {
				uint64_t ts = doca_gpu_dev_eth_bswap64(cqe64->timestamp);
				out_attr[packet_idx].timestamp_ns =
					(ts & 0xffffffff) + (ts >> 32) * DOCA_GPUNETIO_ETH_NS_PER_S;
			}

			packet_idx++;
		}

		if (timeout_ns > 0) {
			rx_now = doca_gpu_dev_eth_query_globaltimer();
			if ((rx_now - rx_start) > timeout_ns)
				break;
		}

		if (max_rx_pkts > 0 && packet_idx >= (max_rx_pkts - 1))
			break;
	}

	if (packet_idx > 0) {
		if (mcst_mode == DOCA_GPUNETIO_ETH_MCST_ENABLED ||
		    (mcst_mode == DOCA_GPUNETIO_ETH_MCST_AUTO && rxq->need_mcst == 1))
			doca_gpu_dev_eth_rxq_mcst<nic_handler>(&(rxq->mcst_qp));

		DOCA_GPUNETIO_ETH_VOLATILE(*out_pkt_num) = packet_idx;

		cqe64 = (volatile struct mlx5_cqe64 *)&cqe[(cqe_ci + packet_idx - 1) & cqe_mask];
		wqe_id_max = DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_id);
		doca_gpu_dev_eth_rxq_submit_cq_dbr(rxq, cqe_ci + packet_idx);

		if (rxq->striding_rq) {
			if (wqe_id_max != rxq->wqe_id_last) {
				if (wqe_id_max > rxq->wqe_id_last)
					doca_gpu_dev_eth_rxq_submit_dbr(rxq,
									DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) +
										(wqe_id_max - rxq->wqe_id_last));
				else
					doca_gpu_dev_eth_rxq_submit_dbr(
						rxq,
						DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) +
							(((DOCA_GPUNETIO_ETH_WQE_PI_MASK + 1) - rxq->wqe_id_last) +
							 wqe_id_max));
				rxq->wqe_id_last = wqe_id_max;
			}
		} else
			doca_gpu_dev_eth_rxq_submit_dbr(rxq, DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) + packet_idx);
	}

	return status;
}

/**
 * @brief Receive packets through the GPU handler of an Ethernet rxq object.
 * This function must be invoked per-warp.
 * It's developer responsibility to ensure each warp invoking this function operates on a different Ethernet receive
 * object. Function will return upon receiving the indicated maximum number of packets or waiting the indicated number
 * of nanoseconds. If timeout is 0, it's ignored and only the maximum number of packets indicates the exit condition.
 *
 * @param [in] rxq
 * GPU handler for Ethernet receive queue.
 * @param [in] max_rx_pkts
 * Max number of packets to receive. Every thread in warp tries to receive at least once so max_rx_pkts should be a
 * multiple of the thread warp number (DOCA_GPUNETIO_ETH_WARP_SIZE). If 0, no limit to the number of packets.
 * @param [in] timeout_ns
 * Max number of nanoseconds to wait before exit from the receive. If 0, no limit to the time spent in function.
 * @param [out] out_first_pkt_idx
 * Index of the first received packet.
 * @param [out] out_pkt_num
 * Total number of received packets.
 * @param [out] out_attr
 * Per-packet attributes. Filled only if rx_attr != DOCA_GPUNETIO_ETH_RX_ATTR_NONE.
 * Caller must ensure the size is great enough to hold stats of all received packets.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 *
 */
template <enum doca_gpu_dev_eth_mcst_mode mcst_mode = DOCA_GPUNETIO_ETH_MCST_DISABLED,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_eth_rxq_attr_type rx_attr = DOCA_GPUNETIO_ETH_RX_ATTR_NONE>
__device__ inline doca_error_t doca_gpu_dev_eth_rxq_recv_warp(struct doca_gpu_eth_rxq *rxq,
							      uint32_t max_rx_pkts,
							      uint64_t timeout_ns,
							      uint64_t *out_first_pkt_idx,
							      uint32_t *out_pkt_num,
							      struct doca_gpu_dev_eth_rxq_attr *out_attr)
{
	__shared__ volatile uint32_t exit_loop[1];

	const uint64_t cqe_ci = DOCA_GPUNETIO_ETH_VOLATILE(rxq->cqe_ci);
	const uint32_t cqe_num = __ldg(&rxq->cqe_num);
	const uint32_t cqe_mask = __ldg(&rxq->cqe_mask);
	const struct mlx5_cqe64 *cqe = (struct mlx5_cqe64 *)__ldg((uintptr_t *)&rxq->cqe_addr);

	doca_error_t status = DOCA_SUCCESS;
	volatile struct mlx5_cqe64 *cqe64;
	uint8_t opown, opcode;
	uint16_t rx_pkts_thread = 0, tmp = 0;
	uint32_t lane_idx = doca_gpu_dev_eth_get_lane_id();
	uint32_t packet_idx = lane_idx;
	uint64_t out_first_pkt_idx_local = DOCA_GPUNETIO_ETH_RX_NO_PKT;
	unsigned long long rx_start = 0, rx_now = 0;

	if (lane_idx == 0) {
		*out_pkt_num = 0;
		*out_first_pkt_idx = DOCA_GPUNETIO_ETH_RX_NO_PKT;
		DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 0;
		if (timeout_ns > 0)
			rx_start = doca_gpu_dev_eth_query_globaltimer();
	}

	if (max_rx_pkts > 0) {
		tmp = max_rx_pkts / DOCA_GPUNETIO_ETH_WARP_SIZE;
		if (tmp == 0)
			tmp = 1;
	}

	__syncwarp();

	while (DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) == 0) {
		cqe64 = (volatile struct mlx5_cqe64 *)&cqe[(cqe_ci + packet_idx) & cqe_mask];
		opown = doca_gpu_dev_eth_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
		opcode = opown >> DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT;
		if ((opcode != MLX5_CQE_INVALID) &&
		    !((opown & MLX5_CQE_OWNER_MASK) ^ !!((cqe_ci + packet_idx) & cqe_num))) {
			if (opcode == (uint8_t)MLX5_CQE_RESP_ERR || opcode == (uint8_t)MLX5_CQE_REQ_ERR) {
				doca_gpu_dev_eth_print_cqe_err(cqe64);
				status = DOCA_ERROR_DRIVER;
				DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 1;
				break;
			}

			// Keep track of the first received packet
			if (packet_idx == 0) {
				if (rxq->striding_rq)
					out_first_pkt_idx_local =
						((DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_id) & rxq->wqe_mask) *
						 rxq->wqe_strides_num) +
						DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_counter);
				else
					out_first_pkt_idx_local = DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_counter) &
								  rxq->wqe_mask;
			}

			if (rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_ALL) {
				if (rxq->striding_rq)
					out_attr[packet_idx].bytes = doca_gpu_dev_eth_bswap32(cqe64->byte_cnt) & 0xFFFF;
				else
					out_attr[packet_idx].bytes = doca_gpu_dev_eth_bswap32(cqe64->byte_cnt);
			}

			if (rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_ALL || rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_TS) {
				uint64_t ts = doca_gpu_dev_eth_bswap64(cqe64->timestamp);
				out_attr[packet_idx].timestamp_ns =
					(ts & 0xffffffff) + (ts >> 32) * DOCA_GPUNETIO_ETH_NS_PER_S;
			}

			packet_idx += DOCA_GPUNETIO_ETH_WARP_SIZE;
			rx_pkts_thread++;
		}

		__syncwarp();

		if (max_rx_pkts > 0 && rx_pkts_thread >= tmp)
			DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 1;

		if (lane_idx == 0 && timeout_ns > 0) {
			rx_now = doca_gpu_dev_eth_query_globaltimer();
			if ((rx_now - rx_start) > timeout_ns)
				DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 1;
		}

		__syncwarp();
	}

#pragma unroll
	for (int offset = 16; offset > 0; offset /= 2)
		rx_pkts_thread += __shfl_down_sync(DOCA_GPUNETIO_ETH_WARP_FULL_MASK, rx_pkts_thread, offset);
	__syncwarp();

	if (lane_idx == 0 && rx_pkts_thread > 0) {
		if (out_first_pkt_idx_local != DOCA_GPUNETIO_ETH_RX_NO_PKT) {
			if (mcst_mode == DOCA_GPUNETIO_ETH_MCST_ENABLED ||
				(mcst_mode == DOCA_GPUNETIO_ETH_MCST_AUTO && rxq->need_mcst == 1))
				doca_gpu_dev_eth_rxq_mcst<nic_handler>(&(rxq->mcst_qp));

			DOCA_GPUNETIO_ETH_VOLATILE(*out_pkt_num) = rx_pkts_thread;
			DOCA_GPUNETIO_ETH_VOLATILE(*out_first_pkt_idx) = out_first_pkt_idx_local;

			cqe64 = (volatile struct mlx5_cqe64 *)&cqe[(cqe_ci + rx_pkts_thread - 1) & cqe_mask];
			tmp = DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_id);

			doca_gpu_dev_eth_rxq_submit_cq_dbr(rxq, cqe_ci + rx_pkts_thread);

			if (rxq->striding_rq) {
				if (tmp != rxq->wqe_id_last) {
					if (tmp > rxq->wqe_id_last)
						doca_gpu_dev_eth_rxq_submit_dbr(rxq,
										DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) +
											(tmp - rxq->wqe_id_last));
					else
						doca_gpu_dev_eth_rxq_submit_dbr(
							rxq,
							DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) +
								(((DOCA_GPUNETIO_ETH_WQE_PI_MASK + 1) - rxq->wqe_id_last) +
								tmp));
					rxq->wqe_id_last = tmp;
				}
			} else
				doca_gpu_dev_eth_rxq_submit_dbr(rxq, DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) + rx_pkts_thread);
		} else
			DOCA_GPUNETIO_ETH_VOLATILE(*out_pkt_num) = 0;
	}

	__syncwarp();

	return status;
}

/**
 * @brief Receive packets through the GPU handler of an Ethernet rxq object.
 * This function must be invoked per-block.
 * It's developer responsibility to ensure each block invoking this function operates on a different Ethernet receive
 * object. Function will return upon receiving the indicated maximum number of packets or waiting the indicated number
 * of nanoseconds. If maximum number of packets is 0, it's ignored and only the timeout indicates the exit condition. If
 * timeout is 0, it's ignored and only the maximum number of packets indicates the exit condition. If both are 0, the
 * function will never return.
 *
 * @param [in] rxq
 * GPU handler for Ethernet receive queue.
 * @param [in] max_rx_pkts
 * Max number of packets to receive. Every thread in block tries to receive at least once so max_rx_pkts should be a
 * multiple of the thread block number. If 0, no limit to the number of packets.
 * @param [in] timeout_ns
 * Max number of nanoseconds to wait before exit from the receive. If 0, no limit to the time spent in function.
 * @param [out] out_first_pkt_idx
 * Index of the first received packet.
 * @param [out] out_pkt_num
 * Total number of received packets.
 * @param [out] out_attr
 * Per-packet attributes. Filled only if rx_attr != DOCA_GPUNETIO_ETH_RX_ATTR_NONE.
 * Caller must ensure the size is great enough to hold stats of all received packets.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - arguments are invalid.
 *
 */
template <enum doca_gpu_dev_eth_mcst_mode mcst_mode = DOCA_GPUNETIO_ETH_MCST_DISABLED,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_eth_rxq_attr_type rx_attr = DOCA_GPUNETIO_ETH_RX_ATTR_NONE>
__device__ inline doca_error_t doca_gpu_dev_eth_rxq_recv_block(struct doca_gpu_eth_rxq *rxq,
							       const uint32_t max_rx_pkts,
							       const uint64_t timeout_ns,
							       uint64_t *out_first_pkt_idx,
							       uint32_t *out_pkt_num,
							       struct doca_gpu_dev_eth_rxq_attr *out_attr)
{
	__shared__ volatile uint32_t exit_loop[1];

	const uint16_t nthreads = blockDim.x * blockDim.y;
	const uint64_t cqe_ci = DOCA_GPUNETIO_ETH_VOLATILE(rxq->cqe_ci);
	const uint32_t cqe_num = __ldg(&rxq->cqe_num);
	const uint32_t cqe_mask = __ldg(&rxq->cqe_mask);
	const struct mlx5_cqe64 *cqe = (struct mlx5_cqe64 *)__ldg((uintptr_t *)&rxq->cqe_addr);

	doca_error_t status = DOCA_SUCCESS;
	volatile struct mlx5_cqe64 *cqe64;
	uint8_t opown, opcode;
	uint16_t rx_pkts_thread = 0, tmp = 0;
	uint32_t packet_idx = threadIdx.x;
	uint64_t out_first_pkt_idx_local = DOCA_GPUNETIO_ETH_RX_NO_PKT;
	unsigned long long rx_start = 0, rx_now = 0;

	if (threadIdx.x == 0) {
		DOCA_GPUNETIO_ETH_VOLATILE(*out_pkt_num) = 0;
		DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 0;
		if (timeout_ns > 0)
			rx_start = doca_gpu_dev_eth_query_globaltimer();
	}

	if (max_rx_pkts > 0) {
		tmp = max_rx_pkts / nthreads;
		if (tmp == 0)
			tmp = 1;
	}

	__syncthreads();

	while (DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) == 0) {
		cqe64 = (volatile struct mlx5_cqe64 *)&cqe[(cqe_ci + packet_idx) & cqe_mask];
		opown = doca_gpu_dev_eth_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
		opcode = opown >> DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT;
		if ((opcode != MLX5_CQE_INVALID) &&
		    !((opown & MLX5_CQE_OWNER_MASK) ^ !!((cqe_ci + packet_idx) & cqe_num))) {
			if (opcode == (uint8_t)MLX5_CQE_RESP_ERR || opcode == (uint8_t)MLX5_CQE_REQ_ERR) {
				doca_gpu_dev_eth_print_cqe_err(cqe64);
				status = DOCA_ERROR_DRIVER;
				DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 1;
				break;
			}

			if (packet_idx == 0) {
				if (rxq->striding_rq)
					out_first_pkt_idx_local =
						((DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_id) & rxq->wqe_mask) *
						 rxq->wqe_strides_num) +
						DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_counter);
				else
					out_first_pkt_idx_local = DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_counter) &
								  rxq->wqe_mask;
			}

			if (rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_ALL) {
				if (rxq->striding_rq)
					out_attr[packet_idx].bytes = doca_gpu_dev_eth_bswap32(cqe64->byte_cnt) & 0xFFFF;
				else
					out_attr[packet_idx].bytes = doca_gpu_dev_eth_bswap32(cqe64->byte_cnt);
			}

			if (rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_ALL || rx_attr == DOCA_GPUNETIO_ETH_RX_ATTR_TS) {
				uint64_t ts = doca_gpu_dev_eth_bswap64(cqe64->timestamp);
				out_attr[packet_idx].timestamp_ns =
					(ts & 0xffffffff) + (ts >> 32) * DOCA_GPUNETIO_ETH_NS_PER_S;
			}

			packet_idx += nthreads;
			rx_pkts_thread++;
		}
		__syncthreads();

		if (max_rx_pkts > 0 && rx_pkts_thread >= tmp)
			DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 1;

		if (threadIdx.x == 0 && timeout_ns > 0) {
			rx_now = doca_gpu_dev_eth_query_globaltimer();
			if ((rx_now - rx_start) > timeout_ns)
				DOCA_GPUNETIO_ETH_VOLATILE(exit_loop[0]) = 1;
		}
		__syncthreads();
	}

#pragma unroll
	for (int offset = 16; offset > 0; offset /= 2)
		rx_pkts_thread += __shfl_down_sync(DOCA_GPUNETIO_ETH_WARP_FULL_MASK, rx_pkts_thread, offset);
	__syncthreads();

	if ((doca_gpu_dev_eth_get_lane_id() == 0) && rx_pkts_thread > 0)
		atomicAdd_block(out_pkt_num, rx_pkts_thread);
	__syncthreads();

	if (threadIdx.x == 0 && *out_pkt_num > 0) {
		if (out_first_pkt_idx_local != DOCA_GPUNETIO_ETH_RX_NO_PKT) {
			if (mcst_mode == DOCA_GPUNETIO_ETH_MCST_ENABLED ||
				(mcst_mode == DOCA_GPUNETIO_ETH_MCST_AUTO && rxq->need_mcst == 1))
				doca_gpu_dev_eth_rxq_mcst<nic_handler>(&(rxq->mcst_qp));

			DOCA_GPUNETIO_ETH_VOLATILE(*out_first_pkt_idx) = out_first_pkt_idx_local;

			cqe64 = (volatile struct mlx5_cqe64 *)&cqe[(cqe_ci + *out_pkt_num - 1) & cqe_mask];
			tmp = DOCA_GPUNETIO_ETH_BSWAP16(cqe64->wqe_id);

			doca_gpu_dev_eth_rxq_submit_cq_dbr(rxq, cqe_ci + *out_pkt_num);

			if (rxq->striding_rq) {
				if (tmp != rxq->wqe_id_last) {
					if (tmp > rxq->wqe_id_last)
						doca_gpu_dev_eth_rxq_submit_dbr(rxq,
										DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) +
											(tmp - rxq->wqe_id_last));
					else
						doca_gpu_dev_eth_rxq_submit_dbr(
							rxq,
							DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) +
								(((DOCA_GPUNETIO_ETH_WQE_PI_MASK + 1) - rxq->wqe_id_last) +
								tmp));
					rxq->wqe_id_last = tmp;
				}
			} else
				doca_gpu_dev_eth_rxq_submit_dbr(rxq, DOCA_GPUNETIO_ETH_VOLATILE(rxq->wqe_pi) + *out_pkt_num);
		} else
			DOCA_GPUNETIO_ETH_VOLATILE(*out_pkt_num) = 0;
	}

	__syncthreads();

	return status;
}

/**
 * @brief Ethernet receiver switch function. Redirect to thread, warp or block scope.
 *
 * @param [in] rxq
 * GPU handler for Ethernet receive queue.
 * @param [in] max_rx_pkts
 * Max number of packets to receive. If 0, no limit to the number of packets.
 * @param [in] timeout_ns
 * Max number of nanoseconds to wait before exit from the receive. If 0, no limit to the time spent in function.
 * @param [out] out_first_pkt_idx
 * Index of the first received packet.
 * @param [out] out_pkt_num
 * Total number of received packets.
 * @param [out] out_attr
 * Per-packet attributes. Filled only if rx_attr != DOCA_GPUNETIO_ETH_RX_ATTR_NONE
 */
template <enum doca_gpu_dev_eth_exec_scope exec_scope = DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD,
	  enum doca_gpu_dev_eth_mcst_mode mcst_mode = DOCA_GPUNETIO_ETH_MCST_DISABLED,
	  enum doca_gpu_dev_eth_nic_handler nic_handler = DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO,
	  enum doca_gpu_dev_eth_rxq_attr_type rx_attr = DOCA_GPUNETIO_ETH_RX_ATTR_NONE>
__device__ inline doca_error_t doca_gpu_dev_eth_rxq_recv(struct doca_gpu_eth_rxq *rxq,
							 uint32_t max_rx_pkts,
							 uint64_t timeout_ns,
							 uint64_t *out_first_pkt_idx,
							 uint32_t *out_pkt_num,
							 struct doca_gpu_dev_eth_rxq_attr *out_attr)
{
	doca_error_t status;
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD)
		status = doca_gpu_dev_eth_rxq_recv_thread<mcst_mode, nic_handler, rx_attr>(rxq,
											   max_rx_pkts,
											   timeout_ns,
											   out_first_pkt_idx,
											   out_pkt_num,
											   out_attr);
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP)
		status = doca_gpu_dev_eth_rxq_recv_warp<mcst_mode, nic_handler, rx_attr>(rxq,
											 max_rx_pkts,
											 timeout_ns,
											 out_first_pkt_idx,
											 out_pkt_num,
											 out_attr);
	if (exec_scope == DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK)
		status = doca_gpu_dev_eth_rxq_recv_block<mcst_mode, nic_handler, rx_attr>(rxq,
											  max_rx_pkts,
											  timeout_ns,
											  out_first_pkt_idx,
											  out_pkt_num,
											  out_attr);
	return status;
}

/**
 * @brief Return packet memory address at a specific index from the buffer associated to a Rxq
 *
 * @param [in] rxq - GPU Rxq Queue Pair (QP)
 * @param [in] packet_idx - Packet index
 * @return packet address
 */
__device__ inline uint64_t doca_gpu_dev_eth_rxq_get_pkt_addr(struct doca_gpu_eth_rxq *rxq, uint64_t packet_idx)
{
	// Avoid expensive modulo operation
	if (packet_idx >= rxq->pkt_num)
		return rxq->pkt_addr + (rxq->max_pkt_sz * (packet_idx - rxq->pkt_num));
	return rxq->pkt_addr + (rxq->max_pkt_sz * packet_idx);
}

/**
 * @brief Return the memory key from the buffer associated to a Rxq
 *
 * @param [in] rxq - GPU Rxq Queue Pair (QP)
 * @return memory mkey
 */
__device__ inline uint32_t doca_gpu_dev_eth_rxq_get_pkt_mkey(struct doca_gpu_eth_rxq *rxq)
{
	return rxq->pkt_mkey;
}

/**
 * @brief Return CQE timestamp associated to the corresponding packet index.
 * This method may report an inaccurate timestamp if called some time after the receive operation:
 * the CQE may be replaced with a newer one in the same position.
 * As an alternative, set rx_attr in the recv operation.
 *
 * @param [in] rxq - GPU Rxq Queue Pair (QP)
 * @param [in] packet_idx - Packet index
 * @return CQE timestamp associated to the corresponding packet index.
 */
__device__ inline uint64_t doca_gpu_dev_eth_rxq_get_pkt_ts(const struct doca_gpu_eth_rxq *rxq, uint64_t packet_idx)
{
	volatile struct mlx5_cqe64 *cqe;

	cqe = &(((volatile struct mlx5_cqe64 *)rxq->cqe_addr)[packet_idx & rxq->cqe_mask]);
	uint64_t ts = doca_gpu_dev_eth_bswap64(cqe->timestamp);
	return ((ts & 0xffffffff) + (ts >> 32) * DOCA_GPUNETIO_ETH_NS_PER_S);
}

/**
 * @brief Return CQE received byte count associated to the corresponding packet index.
 * This method may report an inaccurate byte value if called some time after the receive operation:
 * the CQE may be replaced with a newer one in the same position.
 * As an alternative, set rx_attr in the recv operation.
 *
 * @param [in] rxq - GPU Rxq Queue Pair (QP)
 * @param [in] packet_idx - Packet index
 * @return CQE received byte count associated to the corresponding packet index.
 */
__device__ inline uint32_t doca_gpu_dev_eth_rxq_get_pkt_bytes(const struct doca_gpu_eth_rxq *rxq, uint64_t packet_idx)
{
	volatile struct mlx5_cqe64 *cqe;

	cqe = &(((volatile struct mlx5_cqe64 *)rxq->cqe_addr)[packet_idx & rxq->cqe_mask]);
	if (rxq->striding_rq)
		return doca_gpu_dev_eth_bswap32(cqe->byte_cnt) & 0xFFFF;
	return doca_gpu_dev_eth_bswap32(cqe->byte_cnt);
}

/**
 * @brief Return CQE associated to the corresponding packet index.
 * This method may report an inaccurate byte value if called some time after the receive operation:
 * the CQE may be replaced with a newer one in the same position.
 *
 * @param [in] rxq - GPU Rxq Queue Pair (QP)
 * @param [in] packet_idx - Packet index
 * @return CQE associated to the corresponding packet index.
 */
__device__ inline struct mlx5_cqe64 *doca_gpu_dev_eth_rxq_get_pkt_cqe(const struct doca_gpu_eth_rxq *rxq,
								      uint64_t packet_idx)
{
	return &(((struct mlx5_cqe64 *)rxq->cqe_addr)[packet_idx & rxq->cqe_mask]);
}

#endif /* DOCA_GPUNETIO_DEVICE_ETH_RXQ_H */

/** @} */
