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
 * @file doca_gpunetio_verbs_def.h
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
#ifndef DOCA_GPUNETIO_VERBS_DEF_H
#define DOCA_GPUNETIO_VERBS_DEF_H

#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <infiniband/mlx5dv.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro to temporarily cast a variable to volatile.
 */
#define DOCA_GPUNETIO_VERBS_VOLATILE(x) (*(volatile typeof(x) *)&(x))

/**
 * Default warp size value of 32 threads
 */
#define DOCA_GPUNETIO_VERBS_WARP_SIZE 32

/**
 * Default warp full mask value
 */
#define DOCA_GPUNETIO_VERBS_WARP_FULL_MASK 0xffffffff

/**
 * Default page size alignment on GPU
 */
#define DOCA_GPUNETIO_VERBS_PAGE_SIZE 65536

/**
 * CQE Consumer Index Mask - 24bits counter
 */
#define DOCA_GPUNETIO_VERBS_CQE_CI_MASK 0xFFFFFF

/**
 * WQE Producer Index Mask - 16bits counter
 */
#define DOCA_GPUNETIO_VERBS_WQE_PI_MASK 0xFFFF

/**
 * SQ WQE Shift - 64B
 */
#define DOCA_GPUNETIO_MLX5_WQE_SQ_SHIFT 6

/**
 * RQ WQE Shift - 16B
 */
#define DOCA_GPUNETIO_MLX5_WQE_RQ_SHIFT 4

/**
 * Set to 1 if mkeys passed to the wqe functions
 * are already swapped by application.
 * Otherwise set it to 0.
 */
#define DOCA_GPUNETIO_VERBS_MKEY_SWAPPED 1

/**
 * Enable debug prints in this headerfile.
 * Bad for performance, should be used only for debugging
 */
#define DOCA_GPUNETIO_VERBS_ENABLE_DEBUG 0

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
#include <assert.h>
/**
 * Assert
 */
#define DOCA_GPUNETIO_VERBS_ASSERT(x) assert(x)
#else
/**
 * Assert empty
 */
#define DOCA_GPUNETIO_VERBS_ASSERT(x) \
	do { \
	} while (0)
#endif

/**
 * WQE data segment inline data with byte count
 */
#define DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE 28

/**
 * CQE Opcode Shift Bytes
 */
#define DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT 4

/**
 * CQE size
 */
#define DOCA_GPUNETIO_VERBS_CQE_SIZE 64

/**
 * WQE idx shift in ctrl segment
 */
#define DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT 8

/**
 * Max RDMA transfer size shift
 */
#define DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT 30

/**
 * Max RDMA transfer size
 */
#define DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE (1ULL << DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT) // 1GiB

/**
 * WQE Opcodes
 */
enum {
	DOCA_GPUNETIO_MLX5_OPCODE_NOP = 0x00,
	DOCA_GPUNETIO_MLX5_OPCODE_SEND_INVAL = 0x01,
	DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE = 0x08,
	DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE_IMM = 0x09,
	DOCA_GPUNETIO_MLX5_OPCODE_SEND = 0x0a,
	DOCA_GPUNETIO_MLX5_OPCODE_SEND_IMM = 0x0b,
	DOCA_GPUNETIO_MLX5_OPCODE_TSO = 0x0e,
	DOCA_GPUNETIO_MLX5_OPCODE_RDMA_READ = 0x10,
	DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_CS = 0x11,
	DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_FA = 0x12,
	DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_MASKED_CS = 0x14,
	DOCA_GPUNETIO_MLX5_OPCODE_ATOMIC_MASKED_FA = 0x15,
	DOCA_GPUNETIO_MLX5_OPCODE_FMR = 0x19,
	DOCA_GPUNETIO_MLX5_OPCODE_LOCAL_INVAL = 0x1b,
	DOCA_GPUNETIO_MLX5_OPCODE_WAIT = 0x0f,
	DOCA_GPUNETIO_MLX5_OPCODE_CONFIG_CMD = 0x1f,
	DOCA_GPUNETIO_MLX5_OPCODE_SET_PSV = 0x20,
	DOCA_GPUNETIO_MLX5_OPCODE_DUMP = 0x23,
	DOCA_GPUNETIO_MLX5_OPCODE_UMR = 0x25,
	DOCA_GPUNETIO_MLX5_OPCODE_TAG_MATCHING = 0x28,
	DOCA_GPUNETIO_MLX5_OPCODE_FLOW_TBL_ACCESS = 0x2c,
	DOCA_GPUNETIO_MLX5_OPCODE_MMO = 0x2F,
};

/**
 * Enum CQE CTRL flags
 */
enum {
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_ON_CQE_ERROR = 0x0,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_ON_FIRST_CQE_ERROR = 0x1,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_ALWAYS = 0x2,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_AND_EQE = 0x3,
};

/**
 * Enum CQE CTRL FM
 */
enum {
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_NO_FENCE = 0x0,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_INITIATOR_SMALL_FENCE = 0x1,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_FENCE = 0x2,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_STRONG_ORDERING = 0x3,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_FENCE_AND_INITIATOR_SMALL_FENCE = 0x4,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_CUSTOM = 0x100, /* None of the previous, use custom value */
};

/**
 * GPUNetIO Verbs flags for WQE control segment
 */
enum doca_gpu_dev_verbs_wqe_ctrl_flags {
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE = DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_ALWAYS << 2,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE = DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_ON_CQE_ERROR << 2,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_FIRST_CQE_ERROR = DOCA_GPUNETIO_MLX5_WQE_CTRL_CE_CQE_ON_FIRST_CQE_ERROR << 2,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_SOLICITED = 1 << 1,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_FENCE = DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_FENCE_AND_INITIATOR_SMALL_FENCE << 5,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE = DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_INITIATOR_SMALL_FENCE << 5,
	DOCA_GPUNETIO_MLX5_WQE_CTRL_STRONG_ORDERING = DOCA_GPUNETIO_MLX5_WQE_CTRL_FM_STRONG_ORDERING << 5
};

/**
 * DBR offsets
 */
enum {
	DOCA_GPUNETIO_MLX5_RCV_DBR = 0,
	DOCA_GPUNETIO_MLX5_SND_DBR = 1,
};

/**
 * @enum doca_gpu_dev_verbs_mem_type
 * @brief Memory type of the buffer.
 */
enum doca_gpu_dev_verbs_mem_type {
	DOCA_GPUNETIO_VERBS_MEM_TYPE_AUTO = 0,	   /**< Automatically select the most performant memory type */
	DOCA_GPUNETIO_VERBS_MEM_TYPE_HOST = 1,	   /**< Allocate resource on host memory */
	DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU = 2,	   /**< Allocate resource on GPU memory */
	DOCA_GPUNETIO_VERBS_MEM_TYPE_MAX = INT_MAX /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_verbs_qp_type
 * @brief Memory type of the buffer.
 */
enum doca_gpu_dev_verbs_qp_type {
	DOCA_GPUNETIO_VERBS_QP_SQ = 0, /**< Use QP SQ */
	DOCA_GPUNETIO_VERBS_QP_RQ      /**< Use QP RQ */
};

/**
 * @enum doca_gpu_dev_verbs_exec_scope
 * @brief Execution scope (thread or warp) for one sided and two sided functions.
 */
enum doca_gpu_dev_verbs_exec_scope {
	DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD = 0, /**< The function is executed by each thread independently. No
						      assumptions on other threads is made. */
	DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP /**< The function is executed by the entire warp. All threads in the warp
					       participate to the function. */
};

/**
 * @enum doca_gpu_dev_verbs_sync_scope
 * @brief Synchronization scope.
 */
enum doca_gpu_dev_verbs_sync_scope {
	DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS = 0,	     /**< System synchronization scope */
	DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU = 1,	     /**< GPU synchronization scope */
	DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA = 2,	     /**< CTA synchronization scope */
	DOCA_GPUNETIO_VERBS_SYNC_SCOPE_MAX = INT_MAX /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_verbs_resource_sharing_mode
 * @brief Resource sharing mode.
 */
enum doca_gpu_dev_verbs_resource_sharing_mode {
	DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE = 0, /**< The resource is exclusive to one CUDA thread */
	DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA = 1, /**< The resource is shared among CUDA threads in the same */
							   /**< CTA */
	DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU = 2, /**< The resource is shared among CUDA threads in the same */
							   /**< GPU */
	DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_MAX = INT_MAX /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_verbs_nic_handler
 * @brief The processor that handles the NIC.
 */
enum doca_gpu_dev_verbs_nic_handler {
	DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO = 0,	   /**< Automatically select the most performant handler */
	DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY = 1,	   /**< CPU Proxy */
	DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB = 2,	   /**< GPU SM, regular DB */
	DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF = 3,	   /**< GPU SM, BlueFlame DB */
	DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR = 4, /**< GPU SM, no need to update the DBREC */
	DOCA_GPUNETIO_VERBS_NIC_HANDLER_TYPE_MAX,	   /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_verbs_mcst_mode
 * @brief How to enable the dump in case or RDMA Get or Recv
 */
enum doca_gpu_dev_verbs_mcst_mode {
	DOCA_GPUNETIO_VERBS_MCST_DISABLED = 0, /**< Post a dump WQE after the Get/Recv */
	DOCA_GPUNETIO_VERBS_MCST_ENABLED = 1,  /**< Don't post a dump WQE after the Get/Recv */
};

/**
 * @enum doca_gpu_dev_verbs_gpu_code_opt
 * @brief GPU code optimization for GDA-KI. They can be combined using bitwise or.
 */
enum doca_gpu_dev_verbs_gpu_code_opt {
	DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT = 0,			 /**< Use default code optimization */
	DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE = (1 << 0), /**< Use store.async.release code */
									 /**< optimization */
	DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_HAS_GDRCOPY = (1 << 1),	 /**< Use store.async.release code */
									 /**< optimization */
	DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_MAX = INT_MAX			 /**< Sentinel value */
};

/**
 * @enum doca_gpu_dev_verbs_signal_op
 * @brief Type of signal operation.
 */
enum doca_gpu_dev_verbs_signal_op {
	DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD = 0, /**< Signal operation - Add */
};

/**
 * @enum doca_gpu_dev_verbs_blocking_mode
 * @brief Type of execution mode: blocking waiting for an event or non-blocking
 */
enum doca_gpu_dev_verbs_blocking_mode {
	DOCA_GPUNETIO_VERBS_BLOCKING_MODE_DISABLED = 0,
	DOCA_GPUNETIO_VERBS_BLOCKING_MODE_ENABLED = 1,
};

/**
 * @enum doca_gpu_dev_verbs_gpu_code_opt
 * @brief GPU code optimization for GDA-KI. They can be combined using bitwise or.
 */
enum {
	DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MIN = 3, /**< WQE data segment, write inline min */
	DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MAX = 4, /**< WQE data segment, write inline max */
	DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_ATOMIC_FA_CAS = 4,	/**< WQE data segment, atomic */
	DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_WAIT = 2
};

/**
 * @typedef doca_gpu_dev_verbs_ticket_t
 * @brief Ticket type used in one-sided APIs.
 */
typedef uint64_t doca_gpu_dev_verbs_ticket_t;

/**
 * Describes GPUNetIO dev WQE crtl segment.
 */
struct doca_gpu_dev_verbs_wqe_ctrl_seg {
	__be32 opmod_idx_opcode; /**< opcode + wqe idx */
	__be32 qpn_ds;		 /**< qp number */
	union {
		struct {
			uint8_t signature; /**< signature */
			uint8_t rsvd[2];   /**< reserved */
			uint8_t fm_ce_se;  /**< fm_ce_se */
		};
		struct {
			__be32 signature_fm_ce_se; /**< all flags in or */
		};
	};

	__be32 imm; /**< immediate */
} __attribute__((__aligned__(8)));

/**
 * Describes GPUNetIO dev WQE wait segment.
 */
struct doca_gpu_dev_verbs_wqe_wait_seg {
	uint32_t resv[2]; /**< reserved */
	__be32 max_index; /**< Index to wait */
	__be32 qpn_cqn;	  /**< CQ number */
} __attribute__((__packed__)) __attribute__((__aligned__(8)));

/**
 * @struct doca_gpu_dev_verbs_addr
 * @brief This structure holds the address and key of a memory region.
 */
struct doca_gpu_dev_verbs_addr {
	uint64_t addr; /**< address */
	uint32_t key;  /**< memory key */
};

/**
 * Describes GPUNetIO dev general WQE.
 */
struct doca_gpu_dev_verbs_wqe {
	union {
		/* Generic inline Data */
		struct {
			uint8_t inl_data[64];
		};

		/* Generic Data */
		struct {
			struct mlx5_wqe_data_seg dseg0;
			struct mlx5_wqe_data_seg dseg1;
			struct mlx5_wqe_data_seg dseg2;
			struct mlx5_wqe_data_seg dseg3;
		};

		/* Read/Write */
		struct {
			struct doca_gpu_dev_verbs_wqe_ctrl_seg rw_cseg;
			struct mlx5_wqe_raddr_seg rw_rseg;
			struct mlx5_wqe_data_seg rw_dseg0;
			struct mlx5_wqe_data_seg rw_dseg1;
		};

		/* Atomic */
		struct {
			struct doca_gpu_dev_verbs_wqe_ctrl_seg at_cseg;
			struct mlx5_wqe_raddr_seg at_rseg;
			struct mlx5_wqe_atomic_seg at_seg;
			struct mlx5_wqe_data_seg at_dseg;
		};

		/* Send */
		struct {
			struct doca_gpu_dev_verbs_wqe_ctrl_seg snd_cseg;
			struct mlx5_wqe_data_seg snd_dseg0;
			struct mlx5_wqe_data_seg snd_dseg1;
			struct mlx5_wqe_data_seg snd_dseg2;
		};
	};
} __attribute__((__aligned__(8)));

/**
 * Describes MLX5 Error CQE - extended version
 */
struct mlx5_err_cqe_ex {
	uint8_t rsvd0[32];	 /**< Reserved */
	__be32 srqn;		 /**< srqn */
	uint8_t rsvd1[16];	 /**< Reserved */
	uint8_t hw_err_synd;	 /**< HW error syndrome */
	uint8_t hw_synd_type;	 /**< HW syndome type */
	uint8_t vendor_err_synd; /**< Vendor error syndome */
	uint8_t syndrome;	 /**< CQE syndome */
	__be32 s_wqe_opcode_qpn; /**< WQE QPN */
	__be16 wqe_counter;	 /**< WQE Counter */
	uint8_t signature;	 /**< Signature */
	uint8_t op_own;		 /**< OP owner */
};

/**
 * Describes MLX5 CQE 32B inline
 */
struct doca_gpu_dev_verbs_cqe64_inline {
	uint8_t inl_data[32];  /**< Inlined data, up to 32B */
	__be32 srqn_uidx;      /**< SRQ number or user index */
	__be32 imm_inval_pkey; /**< Immediate value */
	uint8_t app;	       /**< TBD */
	uint8_t app_op;	       /**< TBD */
	__be16 app_info;       /**< TBD */
	__be32 byte_cnt;       /**< Byte received/read */
	__be64 timestamp;      /**< CQE creation timestamp */
	__be32 sop_drop_qpn;   /**< WQE QPN */
	__be16 wqe_counter;    /**< WQE Counter */
	uint8_t signature;     /**< Signature */
	uint8_t op_own;	       /**< OP owner */
};

/**
 * Describes GPUNetIO dev CQ
 */
struct doca_gpu_dev_verbs_cq {
	uint8_t *cqe_daddr;			   /**< CQE address */
	uint32_t cq_num;			   /**< CQ number */
	uint32_t cqe_num;			   /**< Total number of CQEs in CQ */
	__be32 *dbrec;				   /**< CQE Doorbell Record */
	uint64_t cqe_ci;			   /**< CQE Consumer Index */
	uint32_t cqe_mask;			   /**< Mask of total number of CQEs in CQ */
	uint8_t cqe_size;			   /**< Single CQE size (64B default) */
	uint64_t cqe_rsvd;			   /**< All previous CQEs are polled */
	enum doca_gpu_dev_verbs_mem_type mem_type; /**< Memory type of the completion queue */
};

/**
 * Describes GPUNetIO dev QP
 */
struct doca_gpu_dev_verbs_qp {
	uint64_t sq_rsvd_index;	       /**< All WQE slots prior to this index are reserved */
	uint64_t sq_ready_index;       /**< All WQE slots prior to this index are ready */
	uint64_t sq_wqe_pi;	       /**< SQ WQE Producer Index */
	uint32_t sq_num;	       /**< SQ number */
	uint32_t sq_num_shift8;	       /**< SQ number << 8 */
	uint32_t sq_num_shift8_be;     /**< SQ number << 8 big endian */
	uint32_t sq_num_shift8_be_1ds; /**< SQ number << 8 big endian, 1 data segment */
	uint32_t sq_num_shift8_be_2ds; /**< SQ number << 8 big endian, 2 data segment */
	uint32_t sq_num_shift8_be_3ds; /**< SQ number << 8 big endian, 3 data segment */
	uint32_t sq_num_shift8_be_4ds; /**< SQ number << 8 big endian, 4 data segment */
	int sq_lock;		       /**< SQ lock */
	uint16_t sq_wqe_num;	       /**< SQ number of WQE */
	uint16_t sq_wqe_mask;	       /**< SQ number of WQE, mask */
	uint8_t *sq_wqe_daddr;	       /**< SQ WQE memory address */
	__be32 *sq_dbrec;	       /**< SQ Doorbell Record */
	uint64_t *sq_db;	       /**< SQ Doorbell Register */

	uint32_t rq_num;	 /**< RQ number */
	uint64_t rq_wqe_pi;	 /**< RQ WQE Producer Index */
	uint32_t rq_wqe_num;	 /**< RQ number of WQE */
	uint32_t rq_wqe_mask;	 /**< RQ number of WQE, mask */
	uint8_t *rq_wqe_daddr;	 /**< RQ WQE memory address */
	__be32 *rq_dbrec;	 /**< RQ Doorbell Record */
	uint32_t rcv_wqe_size;	 /**< Recv WQE size */
	uint64_t rq_rsvd_index;	 /**< All previous recv WQEs are reserved */
	uint64_t rq_ready_index; /**< All previous recv WQEs are ready */
	int rq_lock;		 /**< RQ lock */

	struct doca_gpu_dev_verbs_cq cq_sq; /**< SQ CQ connected to QP */
	struct doca_gpu_dev_verbs_cq cq_rq; /**< RQ CQ connected to QP */

	enum doca_gpu_dev_verbs_nic_handler nic_handler; /**< NIC handler */
	enum doca_gpu_dev_verbs_mem_type mem_type;	 /**< Memory type of the completion */

	bool need_mcst;		    /**< dump mechanism is needed by this GPU QP */
	uint64_t last_get_wqe_idx;  /**< Last get RMDA Get WQE posted index */
	uint64_t last_dump_wqe_idx; /**< Last get RMDA Dump WQE posted index */
	uint64_t last_recv_wqe_idx; /**< Last get RMDA Recv WQE posted index */

} __attribute__((__aligned__(8)));

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_VERBS_DEF_H */

/** @} */
