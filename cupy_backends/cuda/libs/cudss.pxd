from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

###############################################################################
# Enum
###############################################################################

cpdef enum:
    # cudssStatus_t
    CUDSS_STATUS_SUCCESS = 0
    CUDSS_STATUS_NOT_INITIALIZED = 1
    CUDSS_STATUS_ALLOC_FAILED = 2
    CUDSS_STATUS_INVALID_VALUE = 3
    CUDSS_STATUS_NOT_SUPPORTED = 4
    CUDSS_STATUS_EXECUTION_FAILED = 5
    CUDSS_STATUS_INTERNAL_ERROR = 6

    # cudssConfigParam_t
    CUDSS_CONFIG_REORDERING_ALG = 0
    CUDSS_CONFIG_FACTORIZATION_ALG = 1
    CUDSS_CONFIG_SOLVE_ALG = 2
    CUDSS_CONFIG_USE_MATCHING = 3             # disabled by default
    CUDSS_CONFIG_MATCHING_ALG = 4             # not used by default
    CUDSS_CONFIG_SOLVE_MODE = 5               # not supported right now 
    CUDSS_CONFIG_IR_N_STEPS = 6
    CUDSS_CONFIG_IR_TOL = 7                   # not supported right now
    CUDSS_CONFIG_PIVOT_TYPE = 8
    CUDSS_CONFIG_PIVOT_THRESHOLD = 9
    CUDSS_CONFIG_PIVOT_EPSILON = 10
    CUDSS_CONFIG_MAX_LU_NNZ = 11              # only for CUDSS_ALG_1 and CUDSS_ALG_2
    CUDSS_CONFIG_HYBRID_MODE = 12             # by default: disabled
    CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT = 13
    CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY = 14# by default: enabled
    CUDSS_CONFIG_HOST_NTHREADS = 15
    CUDSS_CONFIG_HYBRID_EXECUTE_MODE = 16     # default: 0 - disabled
    CUDSS_CONFIG_PIVOT_EPSILON_ALG = 17
    CUDSS_CONFIG_ND_NLEVELS = 18
    CUDSS_CONFIG_UBATCH_SIZE = 19             # "U" - stands for Uniform
    CUDSS_CONFIG_UBATCH_INDEX = 20
    CUDSS_CONFIG_USE_SUPERPANELS = 21         # by default: enabled
    CUDSS_CONFIG_DEVICE_COUNT = 22
    CUDSS_CONFIG_DEVICE_INDICES = 23
    CUDSS_CONFIG_SCHUR_MODE = 24
    CUDSS_CONFIG_DETERMINISTIC_MODE = 25

    # cudssDataParam_t
    CUDSS_DATA_INFO = 0                  # (out)
    CUDSS_DATA_LU_NNZ = 1                # (out)
    CUDSS_DATA_NPIVOTS = 2               # (out)
    CUDSS_DATA_INERTIA = 3               
    # (out, non-trivial for non-positive-definite matrices)
    CUDSS_DATA_PERM_REORDER_ROW = 4      # (out)
    CUDSS_DATA_PERM_REORDER_COL = 5      # (out)
    CUDSS_DATA_PERM_ROW = 6              # (out,
    # supported only for CUDSS_ALG_1 and CUDSS_ALG_2 reordering algorithms)
    CUDSS_DATA_PERM_COL = 7              # (out,
    # supported only for CUDSS_ALG_1 and CUDSS_ALG_2 reordering algorithms)
    CUDSS_DATA_DIAG = 8                  # (out)
    CUDSS_DATA_USER_PERM = 9             # (in, out)
    # for the user to provide a permutation or retrieve the provided permutation
    CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN = 10
    CUDSS_DATA_COMM = 11                 # (in) communicator
    CUDSS_DATA_MEMORY_ESTIMATES = 12
    CUDSS_DATA_PERM_MATCHING = 13        # (out, supported only when matching is enabled)
    CUDSS_DATA_SCALE_ROW = 14            # (out, supported only when matching is enabled)
    CUDSS_DATA_SCALE_COL = 15            
    # (out, supported only when matching with scaling is enabled)
    CUDSS_DATA_NSUPERPANELS = 16         # (out)
    CUDSS_DATA_USER_SCHUR_INDICES = 17   
    # (in) indices[n] (1s for Schur complement, 0s for the rest)
    CUDSS_DATA_SCHUR_SHAPE = 18          # (out) int64_t shape[3]
    CUDSS_DATA_SCHUR_MATRIX = 19         # (out) cudssMatrix_t
    # (in, out) for the user to provide
    # or retrieve the auxiliary elimination tree information
    CUDSS_DATA_USER_ELIMINATION_TREE = 20
    CUDSS_DATA_ELIMINATION_TREE = 21     # (out) retrieves the elimination tree information
    CUDSS_DATA_USER_HOST_INTERRUPT = 22

    # cudssPhase_t
    CUDSS_PHASE_REORDERING             = 1   # 1 << 0
    CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 2   # 1 << 1
    CUDSS_PHASE_ANALYSIS               = 3
            # CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION
    CUDSS_PHASE_FACTORIZATION          = 4   # 1 << 2
    CUDSS_PHASE_REFACTORIZATION        = 8   # 1 << 3
    CUDSS_PHASE_SOLVE_FWD_PERM         = 16  # 1 << 4
    CUDSS_PHASE_SOLVE_FWD              = 32  # 1 << 5
    CUDSS_PHASE_SOLVE_DIAG             = 64  # 1 << 6
    CUDSS_PHASE_SOLVE_BWD              = 128 # 1 << 7
    CUDSS_PHASE_SOLVE_BWD_PERM         = 256 # 1 << 8
    CUDSS_PHASE_SOLVE_REFINEMENT       = 512 # 1 << 9
    CUDSS_PHASE_SOLVE                  = 1008
            #   CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD
            # | CUDSS_PHASE_SOLVE_DIAG | CUDSS_PHASE_SOLVE_BWD
            # | CUDSS_PHASE_SOLVE_BWD_PERM | CUDSS_PHASE_SOLVE_REFINEMENT

    # cudssMatrixFormat_t
    CUDSS_MFORMAT_DENSE       = 1
    CUDSS_MFORMAT_CSR         = 2
    CUDSS_MFORMAT_BATCH       = 4
    CUDSS_MFORMAT_DISTRIBUTED = 8

    # cudssMatrixType_t
    CUDSS_MTYPE_GENERAL = 0
    CUDSS_MTYPE_SYMMETRIC = 1
    CUDSS_MTYPE_HERMITIAN = 2
    CUDSS_MTYPE_SPD = 3
    CUDSS_MTYPE_HPD = 4

    # cudssMatrixViewType_t
    CUDSS_MVIEW_FULL = 0
    CUDSS_MVIEW_LOWER = 1
    CUDSS_MVIEW_UPPER = 2

    # cudssIndexBase_t
    CUDSS_BASE_ZERO = 0
    CUDSS_BASE_ONE = 1

    # cudssLayout_t
    CUDSS_LAYOUT_COL_MAJOR = 0
    CUDSS_LAYOUT_ROW_MAJOR = 1

    # cudssAlgType_t
    CUDSS_ALG_DEFAULT = 0
    CUDSS_ALG_1 = 1
    CUDSS_ALG_2 = 2
    CUDSS_ALG_3 = 3
    CUDSS_ALG_4 = 4
    CUDSS_ALG_5 = 5

    # cudssPivotType_t
    CUDSS_PIVOT_COL = 0
    CUDSS_PIVOT_ROW = 1
    CUDSS_PIVOT_NONE = 2

    # cudssOpType_t
    CUDSS_SUM = 0
    CUDSS_MAX = 1
    CUDSS_MIN = 2
