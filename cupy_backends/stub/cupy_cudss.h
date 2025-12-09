// Stub header file for cuDSS

#ifndef INCLUDE_GUARD_STUB_CUPY_CUDSS_H
#define INCLUDE_GUARD_STUB_CUPY_CUDSS_H

#define CUDSS_VERSION -1

extern "C" {

    typedef enum {
        CUDSS_STATUS_SUCCESS=0,
    } cudssStatus_t;

    typedef enum {} cudaDataType;
    typedef enum {} cudssConfigParam_t;
    typedef enum {} cudssDataParam_t;
    typedef enum {} cudssPhase_t;
    typedef enum {} cudssMatrixType_t;
    typedef enum {} cudssMatrixViewType_t;
    typedef enum {} cudssIndexBase_t;
    typedef enum {} cudssLayout_t;
    typedef enum {} cudssAlgType_t;
    typedef enum {} cudssPivotType_t;
    typedef enum {} cudssMatrixFormat_t;

    typedef void* cudaStream_t;
    typedef void* cudssHandle_t;;
    typedef void* cudssMatrix_t;
    typedef void* cudssData_t;
    typedef void* cudssConfig_t;

    typedef struct {} cudssDeviceMemHandler_t;

    typedef struct {} cudssDistributedInterface_t;
    typedef enum {} cudssOpType_t;

    typedef struct {} cudssThreadingInterface_t;
    typedef enum {} cudssOpType_t;


    cudssStatus_t cudssConfigSet(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssConfigGet(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssDataSet(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssDataGet(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Main cuDSS routine

    cudssStatus_t cudssExecute(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Setting the stream (in the library handle)

    cudssStatus_t cudssSetStream(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Setting the communication layer library name (in the library handle)

    cudssStatus_t cudssSetCommLayer(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Setting the threading layer library name (in the library handle)

    cudssStatus_t cudssSetThreadingLayer(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Create/Destroy APIs (allocating structures + set defaults)

    cudssStatus_t cudssConfigCreate(...) {
        return CUDSS_STATUS_SUCCESS;
    }
    cudssStatus_t cudssConfigDestroy(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssDataCreate(...) {
        return CUDSS_STATUS_SUCCESS;
    }
    cudssStatus_t cudssDataDestroy(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssCreate(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssCreateMg(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssDestroy(...) {
        return CUDSS_STATUS_SUCCESS;
    }


    // Versioning

    cudssStatus_t cudssGetProperty(...) {
        return CUDSS_STATUS_SUCCESS;
    }


    // Create/Destroy API helpers for matrix wrappers

    cudssStatus_t cudssMatrixCreateDn(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixCreateCsr(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixCreateBatchDn(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixCreateBatchCsr(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixDestroy(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Setters/Getters API helpers for matrix wrappers

    cudssStatus_t cudssMatrixGetDn(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixGetCsr(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixSetValues(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixSetCsrPointers(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixGetBatchDn(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixGetBatchCsr(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixSetBatchValues(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixSetBatchCsrPointers(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixGetFormat(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixSetDistributionRow1d(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssMatrixGetDistributionRow1d(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    // Memory allocator API

    cudssStatus_t cudssGetDeviceMemHandler(...) {
        return CUDSS_STATUS_SUCCESS;
    }

    cudssStatus_t cudssSetDeviceMemHandler(...) {
        return CUDSS_STATUS_SUCCESS;
    }


} // extern "C"


#endif  // INCLUDE_GUARD_STUB_CUPY_CUDSS_H