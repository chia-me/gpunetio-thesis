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
 * @file doca_gpunetio_eth_def.h
 * @page DOCA GPUNetIO Ethernet Device - Definitions
 * @defgroup DOCA_GPUNETIO_ETH_DEF DOCA GPUNetIO Ethernet Device - Definitions
 * @ingroup DOCA_GPUNETIO
 * DOCA GPUNetio device library header to be included in CUDA .cu files.
 * All functions listed here must be called from a GPU CUDA kernel, they won't work from CPU.
 * All functions listed here should be considered as experimental.
 * For more details please refer to the user guide on DOCA devzone.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_ETH_DEF_H
#define DOCA_GPUNETIO_ETH_DEF_H

#include <stdio.h>
#include <stdint.h>
#include <infiniband/mlx5dv.h>
#include <doca_gpunetio.h>
#include <doca_error.h>
#include <doca_types.h>

/**
 * NIC L3 checksum flag value
 */
#define DOCA_GPUNETIO_ETH_WQE_L3_CSUM (1 << 6)

/**
 * NIC L4 checksum flag value
 */
#define DOCA_GPUNETIO_ETH_WQE_L4_CSUM (1 << 7)

/**
 * Default warp size value of 32 threads
 */
#define DOCA_GPUNETIO_ETH_WARP_SIZE 32

/**
 * Default warp full mask value
 */
#define DOCA_GPUNETIO_ETH_WARP_FULL_MASK 0xffffffff

/**
 * Default page size alignment on GPU
 */
#define DOCA_GPUNETIO_ETH_PAGE_SIZE 65536

/**
 * WQE Producer Index Mask - 16bits counter
 */
#define DOCA_GPUNETIO_ETH_WQE_PI_MASK 0xFFFF

/**
 * CQE Consumer Index Mask - 24bits counter
 */
#define DOCA_GPUNETIO_ETH_CQE_CI_MASK 0xFFFFFF

/**
 * CQE Opcode Shift Bytes
 */
#define DOCA_GPUNETIO_ETH_MLX5_CQE_OPCODE_SHIFT 4

/**
 * CQE Size
 */
#define DOCA_GPUNETIO_ETH_CQE_SIZE 64

/**
 * Ctrl segment WQE idx shift
 */
#define DOCA_GPUNETIO_ETH_WQE_IDX_SHIFT 8

/**
 * MLX5 WQE Cltr Opcode Send
 */
#define DOCA_GPUNETIO_ETH_MLX5_OPCODE_SEND 0x0a
/**
 * MLX5 WQE Cltr Opcode Wait Time
 */
#define DOCA_GPUNETIO_ETH_MLX5_OPCODE_WAIT 0xf
/**
 * MLX5 WQE Cltr Opcode Dump
 */
#define DOCA_GPUNETIO_ETH_MLX5_OPCODE_DUMP 0x23
/**
 * MLX5 WQE Cltr Opcode Enhanced Multi-Packet WQE
 */
#define DOCA_GPUNETIO_ETH_MLX5_OPCODE_ENHANCED_MPSW 0x29u
/**
 * MLX5 WQE Cltr Opcode Wait Time Mode
 */
#define DOCA_GPUNETIO_ETH_MLX5_OPC_MOD_WAIT_TIME 0x2

/**
 * SQ WQE Shift - 64B
 */
#define DOCA_GPUNETIO_ETH_MLX5_WQE_SHIFT 6

/**
 * Nanoseconds per second
 */
#define DOCA_GPUNETIO_ETH_NS_PER_S 1000000000

/**
 * Enable debug prints in this headerfile.
 * Bad for performance, should be used only for debugging
 */
#define DOCA_GPUNETIO_ETH_ENABLE_DEBUG 0

#if DOCA_GPUNETIO_ETH_ENABLE_DEBUG == 1
#include <assert.h>
/**
 * Assert
 */
#define DOCA_GPUNETIO_ETH_ASSERT(x) assert(x)
#else
/**
 * Assert empty
 */
#define DOCA_GPUNETIO_ETH_ASSERT(x) \
	do { \
	} while (0)
#endif

/**
 * Swap 16 bit variable
 */
#define DOCA_GPUNETIO_ETH_BSWAP16(v) \
	((((uint16_t)(v) & (UINT16_C(0x00ff))) << 8) | (((uint16_t)(v) & (UINT16_C(0xff00))) >> 8))

/**
 * @enum doca_gpu_eth_send_flags
 * @brief Send WQE flags: receive a CQE notification if send WQE is successful (DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY)
 * or receive just a CQE notification in case of send WQE error (DOCA_GPUNETIO_ETH_SEND_FLAG_NONE).
 * With DOCA_GPUNETIO_ETH_SEND_FLAG_NONE, CQE polling is optional.
 */
enum doca_gpu_eth_send_flags {
	DOCA_GPUNETIO_ETH_SEND_FLAG_NONE = 0,	     /**< No specific property must be enabled.
						      *  Best configuration to enhance the performance.
						      */
	DOCA_GPUNETIO_ETH_SEND_FLAG_NOTIFY = 1 << 0, /**< Return a notification when a send or wait item is actually
						      * executed. Please note this is requires the DOCA PE on the CPU
						      * side to check the Eth Txq context (send queue) and cleanup every
						      * notification (can be set via
						      * doca_eth_gpu_event_notify_send_packet_register function). For
						      * this reason, this flag should be set mostly for debugging
						      * purposes as it may affect the overall performance if CPU is not
						      * fast enough to query and cleanup the notifications.
						      */
};

/**
 * @enum doca_gpu_dev_eth_wait_flags
 * @brief CQE polling wait mode, blocking or not blocking
 */
enum doca_gpu_dev_eth_wait_flags {
	DOCA_GPUNETIO_ETH_WAIT_FLAG_NB = 0, /**< Non-Blocking mode: the wait completion function
					     * doca_gpu_dev_eth_wait_completion checks if the input number of send
					     * operations completed (data has been sent) and exit from the function. If
					     * nothing has been sent, the function doesn't block the execution.
					     */
	DOCA_GPUNETIO_ETH_WAIT_FLAG_B = 1,  /**< Blocking mode: the wait completion function
					     * doca_gpu_dev_eth_wait_completion blocks the execution waiting for all
					     * the  input send operations to be completed.
					     */
};

/**
 * @enum doca_gpu_dev_eth_exec_scope
 * @brief Execution scope (thread or warp) for one sided and two sided functions.
 */
enum doca_gpu_dev_eth_exec_scope {
	DOCA_GPUNETIO_ETH_EXEC_SCOPE_THREAD = 0, /**< The function is executed by each thread independently. No
						      assumptions on other threads is made. */
	DOCA_GPUNETIO_ETH_EXEC_SCOPE_WARP,  /**< The function is executed by the entire warp. All threads in the warp
						participate to the function. */
	DOCA_GPUNETIO_ETH_EXEC_SCOPE_BLOCK, /**< The function is executed by the entire block. All threads in the block
					       participate to the function. */
};

/**
 * @enum doca_gpu_dev_eth_sync_scope
 * @brief Synchronization scope.
 */
enum doca_gpu_dev_eth_sync_scope {
	DOCA_GPUNETIO_ETH_SYNC_SCOPE_SYS = 0,	   /**< System synchronization scope */
	DOCA_GPUNETIO_ETH_SYNC_SCOPE_GPU = 1,	   /**< GPU synchronization scope */
	DOCA_GPUNETIO_ETH_SYNC_SCOPE_CTA = 2,	   /**< CTA synchronization scope */
	DOCA_GPUNETIO_ETH_SYNC_SCOPE_MAX = INT_MAX /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_eth_resource_sharing_mode
 * @brief Resource sharing mode.
 */
enum doca_gpu_dev_eth_resource_sharing_mode {
	DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_EXCLUSIVE = 0, /**< The resource is exclusive to one CUDA thread */
	DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_CTA = 1, /**< The resource is shared among CUDA threads in the same */
							 /**< CTA */
	DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_GPU = 2, /**< The resource is shared among CUDA threads in the same */
							 /**< GPU */
	DOCA_GPUNETIO_ETH_RESOURCE_SHARING_MODE_MAX = INT_MAX /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_eth_nic_handler
 * @brief The processor that handles the NIC.
 */
enum doca_gpu_dev_eth_nic_handler {
	DOCA_GPUNETIO_ETH_NIC_HANDLER_AUTO = 0,	     /**< Automatically select the most performant handler */
	DOCA_GPUNETIO_ETH_NIC_HANDLER_CPU_PROXY = 1, /**< CPU Proxy */
	DOCA_GPUNETIO_ETH_NIC_HANDLER_GPU_SM_DB = 2, /**< GPU SM, regular DB */
	DOCA_GPUNETIO_ETH_NIC_HANDLER_TYPE_MAX,	     /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_eth_cq_poll_mode
 * @brief How to poll the CQ in doca_gpu_dev_eth_txq_poll_completion()
 */
enum doca_gpu_dev_eth_cq_poll_mode {
	DOCA_GPUNETIO_ETH_CQ_POLL_ALL = 0,  /**< Poll all CQEs in input range */
	DOCA_GPUNETIO_ETH_CQ_POLL_LAST = 1, /**< Poll only last CQEs in input range */
};

/**
 * @enum doca_gpu_dev_eth_mcst_mode
 * @brief How to enable the memory consistency for Rxq
 */
enum doca_gpu_dev_eth_mcst_mode {
	DOCA_GPUNETIO_ETH_MCST_DISABLED = 0, /**< Disable Rxq Dump */
	DOCA_GPUNETIO_ETH_MCST_ENABLED,	     /**< Enabled Rxq Dump */
	DOCA_GPUNETIO_ETH_MCST_AUTO,	     /**< Let library decide if Dump is needed */
};

/**
 * @struct doca_gpu_eth_txq
 * @brief GPUNetIO Ethernet handler for a Tx Queue Pair (QP)
 */
struct doca_gpu_eth_txq {
	uint8_t enable_l3_cs_offload; /**< Enable NIC checksum L3 offload */
	uint8_t enable_l4_cs_offload; /**< Enable NIC checksum L4 offload */
	uint64_t sq_rsvd_index;	      /**< All WQE slots prior to this index are reserved */
	uint64_t sq_ready_index;      /**< All WQE slots prior to this index are ready */
	int lock;		      /**< SQ lock */

	uintptr_t wqe_addr;   /**< WQEs buffer address */
	uintptr_t wqe_db_rec; /**< SQ doorbell record address */
	uintptr_t wqe_db;     /**< SQ doorbell */
	uint64_t wqe_pi;      /**< producer index */
	uint16_t wqe_mask;    /**< SQ size mask */

	uint64_t cqe_ci;	/**< consumer index */
	uint32_t cqe_num;	/**< CQ size */
	uint32_t cqe_mask;	/**< CQ size mask */
	uintptr_t cqe_addr;	/**< CQEs buffer address */
	uintptr_t cq_db_rec;	/**< CQ umem doorbell record offset */
	uint64_t cq_rsvd_index; /**< CQ slot atomically reserved index */
	int cq_lock;		/**< CQ lock */

	uint32_t sq_num_shift8;		    /**< SQ id shifted */
	uint32_t sq_num_shift8_be;	    /**< SQ id shifted big-endian */
	uint32_t sq_num_shift8_be_1ds;	    /**< Send WQE pre-defined cflags, 1 segment */
	uint32_t sq_num_shift8_be_2ds;	    /**< Send WQE pre-defined cflags, 2 segments */
	uint32_t const_wait_be;		    /**< Send WQE pre-defined sq num + data segments */
	uint32_t const_wait_wqe_op_be;	    /**< Send WQE pre-defined wait on time op value */
	uint32_t const_cflags_first_err_be; /**< Send WQE pre-defined cflags, CQE first error */
	uint32_t const_cflags_always_be;    /**< Send WQE pre-defined cflags, CQE always */

	/* shared memory info */
	uintptr_t shared_mem_addr_cpu; /**< Shared memory address on CPU */
	uintptr_t shared_mem_addr_gpu; /**< Shared memory address on GPU */

	enum doca_gpu_dev_eth_nic_handler nic_handler; /**< Type of NIC handler: AUTO, DB or CPU Proxy */
};

/**
 * @struct doca_gpu_eth_mcst_qp
 * @brief GPUNetIO handler for a Dump Queue Pair (QP)
 */
struct doca_gpu_eth_mcst_qp {
	uint64_t pkt_addr; /**< GPU memory address */
	uint64_t pkt_num;  /**< GPU memory key */

	uintptr_t wqe_addr;   /**< WQEs buffer address */
	uintptr_t wqe_db_rec; /**< SQ doorbell record address */
	uint64_t wqe_db;      /**< SQ doorbell */
	uint64_t wqe_pi;      /**< producer index */
	uint16_t wqe_mask;    /**< SQ size mask */

	uint64_t cqe_ci;     /**< consumer index */
	uint32_t cqe_num;    /**< CQ size */
	uint32_t cqe_mask;   /**< CQ size mask */
	uintptr_t cqe_addr;  /**< CQEs buffer address */
	uintptr_t cq_db_rec; /**< CQ umem doorbell record offset */

	uintptr_t shared_mem_addr_cpu; /**< Shared memory address on CPU */
	uintptr_t shared_mem_addr_gpu; /**< Shared memory address on GPU */

	enum doca_gpu_dev_eth_nic_handler nic_handler; /**< Type of NIC handler: AUTO, DB or CPU Proxy */
};

/**
 * @struct doca_gpu_eth_rxq
 * @brief GPUNetIO Ethernet handler for a Rx Queue Pair (QP)
 */
struct doca_gpu_eth_rxq {
	uint64_t pkt_addr;   /**< Packets buffer address */
	uint32_t pkt_mkey;   /**< Packets buffer memory key */
	uint64_t pkt_num;    /**< Packets buffer total number */
	uint64_t max_pkt_sz; /**< Packets buffer fixed max size */
	uint8_t striding_rq; /**< Striding QP enabled/disabled */
	bool need_mcst;	     /**< Rxq needs mcst mechanism */

	uintptr_t wqe_addr;	  /**< WQEs buffer address */
	uintptr_t wqe_db_rec;	  /**< SQ doorbell record address */
	uint64_t wqe_pi;	  /**< producer index */
	uint32_t wqe_num;	  /**< SQ number of wqe */
	uint32_t wqe_mask;	  /**< SQ number of wqe, mask */
	uint32_t wqe_strides_num; /**< Number of strides per WQE */
	uint32_t wqe_id_last;	  /**< Last WQE used with striding RQ */
	uint32_t wqe_id_update;	  /**< Update next striding WQE */

	uint64_t cqe_ci;     /**< consumer index */
	uint32_t cqe_num;    /**< CQ number of CQEs */
	uint32_t cqe_mask;   /**< CQ number of CQEs, mask */
	uintptr_t cqe_addr;  /**< CQEs buffer address */
	uintptr_t cq_db_rec; /**< CQ umem doorbell record offset */

	struct doca_gpu_eth_mcst_qp mcst_qp; /**< Dump QP associated to the Rx QP */
};

#endif /**< DOCA_GPUNETIO_VERBS_DEF_H */

/** @} */
