"""Thin wrapper for cuDSS"""

cimport cython  # NOQA

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport int32_t, uint32_t, int64_t, intptr_t

from cupy_backends.cuda cimport stream as stream_module
from cupy_backends.cuda.api cimport runtime

###############################################################################
# Types
###############################################################################
cdef extern from *:
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef enum libraryPropertyType_t 'libraryPropertyType_t':
        MAJOR_VERSION = 0
        MINOR_VERSION = 1
        PATCH_LEVEL   = 2

cdef extern from '../../cupy_cudss.h' nogil:
    ctypedef int cudaDataType_t 'cudaDataType'

    cdef struct cudssContext: 
        pass

    ctypedef cudssContext* cudssHandle_t 'cudssHandle_t'

    cdef struct cudssMatrix: 
        pass

    ctypedef cudssMatrix cudssMatrix_t 'cudssMatrix_t'

    cdef struct cudssConfig: 
        pass

    ctypedef cudssConfig* cudssConfig_t 'cudssConfig_t'

    cdef struct cudssData: 
        pass

    ctypedef cudssData* cudssData_t 'cudssData_t'

    # Non-opaque Data Structures
    ctypedef struct cudssDeviceMemHandler_t:
        pass
    
    # Communication Layer Types
    ctypedef struct cudssDistributedInterface_t:
        pass

    # Threading Layer Types
    ctypedef struct cudssThreadingInterface_t:
        pass

    # Enumerators
    ctypedef int cudssStatus_t 'cudssStatus_t'
    ctypedef int cudssConfigParam_t 'cudssConfigParam_t'
    ctypedef int cudssDataParam_t 'cudssDataParam_t'
    ctypedef int cudssPhase_t 'cudssPhase_t'
    ctypedef int cudssMatrixFormat_t 'cudssMatrixFormat_t'
    ctypedef int cudssMatrixType_t 'cudssMatrixType_t'
    ctypedef int cudssMatrixViewType_t 'cudssMatrixViewType_t'
    ctypedef int cudssIndexBase_t 'cudssIndexBase_t'
    ctypedef int cudssLayout_t 'cudssLayout_t'
    ctypedef int cudssAlgType_t 'cudssAlgType_t'
    ctypedef int cudssPivotType_t 'cudssPivotType_t'
    ctypedef int cudssOpType_t 'cudssOpType_t'

    # Library Management Functions
    cudssStatus_t cudssCreate(cudssHandle_t *handle)
    cudssStatus_t cudssCreateMg(cudssHandle_t *handle,
        int device_count, int *device_indices)
    cudssStatus_t cudssDestroy(cudssHandle_t handle)
    cudssStatus_t cudssGetProperty(
        libraryPropertyType_t propertyType, int *value)
    
    cudssStatus_t cudssSetStream(cudssHandle_t handle, cudaStream_t stream)
    cudssStatus_t cudssSetDeviceMemHandler(cudssHandle_t handle,
        const cudssDeviceMemHandler_t *handler)
    cudssStatus_t cudssGetDeviceMemHandler(cudssHandle_t handle,
        cudssDeviceMemHandler_t *handler)
    cudssStatus_t cudssSetCommLayer(cudssHandle_t handle,
        const char *commLibFileName)
    cudssStatus_t cudssSetThreadingLayer(cudssHandle_t handle,
        const char *thrLibFileName)
    cudssStatus_t cudssConfigCreate(cudssConfig_t *config)
    cudssStatus_t cudssConfigDestroy(cudssConfig_t config)
    cudssStatus_t cudssConfigSet(cudssConfig_t config,
        cudssConfigParam_t param, void *value, size_t sizeInBytes)
    cudssStatus_t cudssConfigGet(cudssConfig_t config,
        cudssConfigParam_t param, void *value,
        size_t sizeInBytes, size_t *sizeWritten)
    cudssStatus_t cudssDataCreate(cudssHandle_t handle, cudssData_t *data)
    cudssStatus_t cudssDataDestroy(cudssHandle_t handle, cudssData_t data)
    cudssStatus_t cudssDataSet(cudssHandle_t handle,
        cudssData_t data, cudssDataParam_t param,
        void *value, size_t sizeInBytes)
    cudssStatus_t cudssDataGet(cudssHandle_t handle,
        cudssData_t data, cudssDataParam_t param,
        void *value, size_t sizeInBytes, size_t *sizeWritten)
    cudssStatus_t cudssExecute(cudssHandle_t handle, int phase,
        cudssConfig_t config, cudssData_t data,
        cudssMatrix_t matrix, cudssMatrix_t solution,
        cudssMatrix_t rhs)
    cudssStatus_t cudssMatrixCreateDn(cudssMatrix_t *matrix,
        int64_t nrows, int64_t ncols, int64_t ld, void *values,
        cudaDataType_t valueType, cudssLayout_t layout)
    cudssStatus_t cudssMatrixCreateBatchDn(cudssMatrix_t *matrix,
        int64_t batchCount, void *nrows, void *ncols, void *ld,
        void **values, cudaDataType_t indexType,
        cudaDataType_t valueType, cudssLayout_t layout)
    cudssStatus_t cudssMatrixCreateCsr(cudssMatrix_t *matrix,
        int64_t nrows, int64_t ncols, int64_t nnz,
        void *rowStart, void *rowEnd, void *colIndices, void *values,
        cudaDataType_t indexType, cudaDataType_t valueType,
        cudssMatrixType_t mtype, cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase)
    cudssStatus_t cudssMatrixCreateBatchCsr(cudssMatrix_t *matrix,
        int64_t batchCount, void *nrows, void *ncols, void *nnz,
    void **rowStart, void **rowEnd, void **colIndices, void **values,
        cudaDataType_t indexType, cudaDataType_t valueType,
        cudssMatrixType_t mtype, cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase)
    cudssStatus_t cudssMatrixDestroy(cudssMatrix_t matrix)
    cudssStatus_t cudssMatrixSetValues(cudssMatrix_t matrix, void *values)
    cudssStatus_t cudssMatrixSetBatchValues(
        cudssMatrix_t matrix, void **values)
    cudssStatus_t cudssMatrixSetCsrPointers(cudssMatrix_t matrix,
        void *rowStart, void *rowEnd, void *colIndices, void *values)
    cudssStatus_t cudssMatrixSetBatchCsrPointers(cudssMatrix_t matrix,
        void **rowStart, void **rowEnd, void **colIndices, void **values)
    cudssStatus_t cudssMatrixSetDistributionRow1d(cudssMatrix_t matrix,
        int64_t first_row, int64_t last_row)
    cudssStatus_t cudssMatrixGetDn(cudssMatrix_t matrix,
        int64_t *nrows, int64_t *ncols, int64_t *ld, void **values,
        cudaDataType_t *valueType, cudssLayout_t *layout)
    cudssStatus_t cudssMatrixGetBatchDn(cudssMatrix_t matrix,
        int64_t *batchCount, void **nrows, void **ncols, void **ld,
        void ***values, cudaDataType_t *indexType,
        cudaDataType_t *valueType, cudssLayout_t *layout)
    cudssStatus_t cudssMatrixGetCsr(cudssMatrix_t matrix,
        int64_t *nrows, int64_t *ncols, int64_t *nnz,
        void **rowStart, void **rowEnd, void **colIndices,
        void **values, cudaDataType_t *indexType,
        cudaDataType_t *valueType, cudssMatrixType_t *mtype,
        cudssMatrixViewType_t *mview, cudssIndexBase_t *indexBase)
    cudssStatus_t cudssMatrixGetBatchCsr(cudssMatrix_t matrix,
        int64_t *batchCount, void **nrows, void **ncols, void **nnz,
        void ***rowStart, void ***rowEnd, void ***colIndices,
        void ***values, cudaDataType_t *indexType,
        cudaDataType_t *valueType, cudssMatrixType_t *mtype,
        cudssMatrixViewType_t *mview, cudssIndexBase_t *indexBase)
    cudssStatus_t cudssMatrixGetDistributionRow1d(
        cudssMatrix_t matrix, int64_t *first_row, int64_t *last_row)
    cudssStatus_t cudssMatrixGetFormat(cudssMatrix_t matrix, int *format)

    # Build-time version
    int CUDSS_VERSION



###############################################################################
# Classes
###############################################################################

cdef class Handle:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cudssHandle_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class DeviceMemHandler:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cudssDeviceMemHandler_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr

cdef class Matrix:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cudssMatrix_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class Data:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cudssData_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr

cdef class Config:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cudssConfig_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr


cdef class DistributedInterface:
    cdef void * _ptr

    def __init__(self):
        self._ptr = PyMem_Malloc(sizeof(cudssDistributedInterface_t))

    def __dealloc__(self):
        PyMem_Free(self._ptr)
        self._ptr = NULL

    @property
    def ptr(self):
        return <intptr_t>self._ptr

###############################################################################
# Error handling
###############################################################################

class CUDSSError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CUDSSError, self).__init__(status)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUDSSError(status)

###############################################################################
# cuDSS: Library Management Functions
###############################################################################

cpdef create(Handle handle):
    """Initializes the cuDSS library handle"""
    status = cudssCreate(<cudssHandle_t*> handle._ptr)
    check_status(status)

cpdef int createMg(Handle handle, int deviceCount) except -1:
    """Initializes the cuDSS library handle for multiple devices"""
    cdef int deviceIndices
    status = cudssCreateMg(<cudssHandle_t*> handle._ptr, deviceCount,
             &deviceIndices)
    check_status(status)
    return deviceIndices

cpdef destroy(Handle handle):
    """Releases hardware resources used by the cuDss library"""
    cdef cudssStatus_t status
    if handle._ptr != NULL:
        status = cudssDestroy(<cudssHandle_t> handle)
        check_status(status)
        handle._ptr = NULL

cpdef int getProperty(libraryPropertyType_t propertyType) except -1:
    cdef int value
    cdef cudssStatus_t status
    with nogil:
        status = cudssGetProperty(propertyType, &value)
    check_status(status)
    return value

cpdef setStream(Handle handle):
    """Sets the stream to be used by the cuDSS library"""
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    status = cudssSetStream(<cudssHandle_t> handle, <runtime.Stream> stream)
    check_status(status)

cpdef setDeviceMemHandler(Handle handle, DeviceMemHandler handler):
    """Set the current device memory handler inside the library handle"""
    status = cudssSetDeviceMemHandler(<cudssHandle_t>handle,
        <cudssDeviceMemHandler_t*>handler._ptr)

cpdef DeviceMemHandler getDeviceMemHandler(Handle handle):
    """Get the current device memory handler"""
    cdef DeviceMemHandler handler = DeviceMemHandler()
    status = cudssGetDeviceMemHandler(<cudssHandle_t>handle,
        <cudssDeviceMemHandler_t*>handler.ptr)
    check_status(status)
    return handler

cpdef setCommLayer(Handle handle, str libFileName):
    """Sets the communication layer to be used in MGMN mode"""
    status = cudssSetCommLayer(<cudssHandle_t>handle,
        libFileName)
    check_status(status)

cpdef setThreadingayer(Handle handle, str thrLibFileName):
    """Sets the threading layer to be used in MGMN mode"""
    status = cudssSetThreadingLayer(<cudssHandle_t>handle,
        thrLibFileName)
    check_status(status)

cpdef configCreate(Config config):
    """Initializes a cuDSS configuration object"""
    status = cudssConfigCreate(<cudssConfig_t*> config._ptr)
    check_status(status)

cpdef configDestroy(Config config):
    """Destroys a cuDSS configuration object"""
    if config._ptr != NULL:
        status = cudssConfigDestroy(<cudssConfig_t> config)
        check_status(status)
        config._ptr = NULL

cpdef configSet(Config config, int param,
                 size_t value, size_t sizeInBytes):
    """Sets a parameter in the cuDSS configuration object"""
    status = cudssConfigSet(<cudssConfig_t>config, <cudssConfigParam_t>param,
        <void *>value, sizeInBytes)
    check_status(status)

cpdef (size_t,size_t) configGet(Config config, int param,
                 size_t sizeInBytes):
    """Gets a parameter from the cuDSS configuration object"""
    cdef size_t value = 0
    cdef size_t sizeWritten
    status = cudssConfigGet(<cudssConfig_t>config, <cudssConfigParam_t>param,
        <void *>value, sizeInBytes, &sizeWritten)
    check_status(status)
    return value, sizeWritten

cpdef Data dataCreate(Handle handle, Data data):
    """Initializes a cuDSS data object"""
    status = cudssDataCreate(<cudssHandle_t> handle,
        <cudssData_t*> data._ptr)
    check_status(status)

cpdef dataDestroy(Handle handle, Data data):
    """Destroys a cuDSS data object"""
    cdef cudssStatus_t status
    if data._ptr != NULL:
        status = cudssDataDestroy(<cudssHandle_t> handle,
                <cudssData_t>data)
        check_status(status)
        data._ptr = NULL

cpdef dataSet(Handle handle, Data data,
                 int param, size_t value, size_t sizeInBytes):
    """Sets a parameter in the cuDSS data object"""
    status = cudssDataSet(<cudssHandle_t> handle,
        <cudssData_t> data, <cudssDataParam_t> param,
        <void *> value, sizeInBytes)
    check_status(status)

cpdef (size_t,size_t) dataGet(Handle handle, Data data,
                 int param, size_t sizeInBytes):
    """Gets a parameter from the cuDSS data object"""
    cdef size_t value = 0
    cdef size_t sizeWritten
    status = cudssDataGet(<cudssHandle_t> handle,
        <cudssData_t> data, <cudssDataParam_t> param,
        <void *> value, sizeInBytes, &sizeWritten)
    check_status(status)
    return value, sizeWritten

cpdef execute(Handle handle, int phase,
                 Config config, Data data,
                 Matrix matrix, Matrix solution,
                 Matrix rhs):
    """Executes a cuDSS computation"""
    status = cudssExecute(<cudssHandle_t> handle, phase,
        <cudssConfig_t> config, <cudssData_t> data,
        <cudssMatrix_t> matrix, <cudssMatrix_t> solution,
        <cudssMatrix_t> rhs)
    check_status(status)

cpdef Matrix matrixCreateDn(Matrix matrix, int64_t nrows,
                    int64_t ncols,int64_t ld, size_t values,
                    int valueType, int layout):
    """Creates a dense matrix object"""
    status = cudssMatrixCreateDn(<cudssMatrix_t*> matrix._ptr,
        nrows, ncols, ld, <void*>values,
        <cudaDataType_t>valueType, <cudssLayout_t>layout)
    check_status(status)

cpdef matrixCreateBatchDn(Matrix matrix, int64_t batchCount,
                          size_t nrows, size_t ncols,
                          size_t ld, size_t values,
                          int indexType, int valueType,
                          int layout):
    """Creates a batched dense matrix object"""
    status = cudssMatrixCreateBatchDn(<cudssMatrix_t*> matrix._ptr,
        batchCount, <void*>nrows, <void*>ncols, <void*>ld,
        <void**>values, <cudaDataType_t>indexType,
        <cudaDataType_t>valueType, <cudssLayout_t>layout)
    check_status(status)

cpdef matrixCreateCsr(Matrix matrix, int64_t nrows,
                        int64_t ncols, int64_t nnz,
                        size_t rowStart, size_t rowEnd,
                        size_t colIndices, size_t values,
                        int indexType, int valueType,
                        int mtype, int mview,
                        int indexBase):
    """Creates a CSR matrix object"""
    status = cudssMatrixCreateCsr(<cudssMatrix_t*> matrix._ptr,
        nrows, ncols, nnz,
        <void*>rowStart, <void*>rowEnd, <void*>colIndices, <void*>values,
        <cudaDataType_t>indexType, <cudaDataType_t>valueType,
        <cudssMatrixType_t>mtype, <cudssMatrixViewType_t>mview,
        <cudssIndexBase_t>indexBase)
    check_status(status)

cpdef matrixCreateBatchCsr(Matrix matrix, int64_t batchCount,
                             size_t nrows, size_t ncols,
                             size_t nnz,
                             size_t rowStart, size_t rowEnd,
                             size_t colIndices, size_t values,
                             int indexType, int valueType,
                             int mtype, int mview,
                             int indexBase):
    """Creates a batched CSR matrix object"""
    status = cudssMatrixCreateBatchCsr(<cudssMatrix_t*> matrix._ptr,
        batchCount, <void*>nrows, <void*>ncols, <void*>nnz,
        <void**>rowStart, <void**>rowEnd, <void**>colIndices, <void**>values,
        <cudaDataType_t>indexType, <cudaDataType_t>valueType,
        <cudssMatrixType_t>mtype, <cudssMatrixViewType_t>mview,
        <cudssIndexBase_t>indexBase)
    check_status(status)

cpdef matrixDestroy(Matrix matrix):
    """Destroys a cuDSS matrix object"""
    cdef cudssStatus_t status
    if matrix._ptr != NULL:
        status = cudssMatrixDestroy(<cudssMatrix_t> matrix)
        check_status(status)
        matrix._ptr = NULL

cpdef matrixSetValues(Matrix matrix, size_t values):
    """Sets the values of a dense matrix"""
    status = cudssMatrixSetValues(<cudssMatrix_t> matrix, <void*>values)
    check_status(status)

cpdef matrixSetBatchValues(Matrix matrix, size_t values):
    """Sets the values of a batched dense matrix"""
    status = cudssMatrixSetBatchValues(<cudssMatrix_t> matrix, <void**>values)
    check_status(status)

cpdef matrixSetCsrPointers(Matrix matrix,
                          size_t rowStart, size_t rowEnd,
                          size_t colIndices, size_t values):
    """Sets the CSR pointers of a CSR matrix""" 
    status = cudssMatrixSetCsrPointers(<cudssMatrix_t> matrix,
        <void*>rowStart, <void*>rowEnd, <void*>colIndices, <void*>values)

cpdef matrixSetBatchCsrPointers(Matrix matrix,
                             size_t rowStart, size_t rowEnd,
                             size_t colIndices, size_t values):
    """Sets the CSR pointers of a batched CSR matrix"""
    status = cudssMatrixSetBatchCsrPointers(<cudssMatrix_t> matrix,
        <void**>rowStart, <void**>rowEnd, <void**>colIndices, <void**>values)
    check_status(status)

cpdef matrixSetDistributionRow1d(Matrix matrix,
                                 int64_t first_row, int64_t last_row):
    """Sets the distribution of a row-1D distributed matrix"""
    status = cudssMatrixSetDistributionRow1d(<cudssMatrix_t> matrix,
        first_row, last_row)
    check_status(status)

cpdef getMatrixGetDn(Matrix matrix):
    """Gets the properties of a dense matrix"""
    cdef int64_t nrows, ncols, ld
    cdef size_t values
    cdef int valueType
    cdef int layout
    status = cudssMatrixGetDn(<cudssMatrix_t> matrix,
        &nrows, &ncols, &ld, <void**>&values,
        <cudaDataType_t*>&valueType, <cudssLayout_t*>&layout)
    check_status(status)
    return nrows, ncols, ld, values, valueType, layout

cpdef getMatrixBatchDn(Matrix matrix):
    """Gets the properties of         pass
a batched dense matrix"""
    cdef int64_t batchCount
    cdef size_t nrows, ncols, ld
    cdef size_t values
    cdef int indexType
    cdef int valueType
    cdef int layout
    status = cudssMatrixGetBatchDn(<cudssMatrix_t> matrix,
        &batchCount, <void**>&nrows, <void**>&ncols, <void**>&ld,
        <void***>&values, <cudaDataType_t*>&indexType,
        <cudaDataType_t*>&valueType, <cudssLayout_t*>&layout)
    check_status(status)
    return batchCount, nrows, ncols, ld, values, indexType, valueType, layout

cpdef getMatrixGetCsr(Matrix matrix):
    """Gets the properties of a CSR matrix"""
    cdef int64_t nrows, ncols, nnz
    cdef size_t rowStart, rowEnd, colIndices, values
    cdef int indexType
    cdef int valueType
    cdef int mtype
    cdef int mview
    cdef int indexBase
    status = cudssMatrixGetCsr(<cudssMatrix_t> matrix,
        &nrows, &ncols, &nnz,
        <void**>&rowStart, <void**>&rowEnd, <void**>&colIndices,
        <void**>&values, <cudaDataType_t*>&indexType,
        <cudaDataType_t*>&valueType, <cudssMatrixType_t*>&mtype,
        <cudssMatrixViewType_t*>&mview, <cudssIndexBase_t*>&indexBase)
    check_status(status)
    return (nrows, ncols, nnz,
        rowStart, rowEnd, colIndices, values,
        indexType, valueType, mtype, mview, indexBase)

cpdef getMatrixGetBatchCsr(Matrix matrix):
    """Gets the properties of a batched CSR matrix"""
    cdef int64_t batchCount
    cdef size_t nrows, ncols, nnz
    cdef size_t rowStart, rowEnd, colIndices, values
    cdef int indexType
    cdef int valueType
    cdef int mtype
    cdef int mview
    cdef int indexBase
    status = cudssMatrixGetBatchCsr(<cudssMatrix_t>matrix,
        &batchCount, <void**>&nrows, <void**>&ncols, <void**>&nnz,
        <void***>&rowStart, <void***>&rowEnd, <void***>&colIndices,
        <void***>&values, <cudaDataType_t*>&indexType,
        <cudaDataType_t*>&valueType, <cudssMatrixType_t*>&mtype,
        <cudssMatrixViewType_t*>&mview, <cudssIndexBase_t*>&indexBase)
    check_status(status)
    return (batchCount, nrows, ncols, nnz,
        rowStart, rowEnd, colIndices, values,
        indexType, valueType, mtype, mview, indexBase)

cpdef getMatrixGetDistributionRow1d(Matrix matrix):
    """Gets the distribution of a row-1D distributed matrix"""
    cdef int64_t first_row, last_row
    status = cudssMatrixGetDistributionRow1d(<cudssMatrix_t> matrix,
        &first_row, &last_row)
    check_status(status)
    return first_row, last_row

cpdef int matrixGetFormat(Matrix matrix) except -1:
    """Gets the format of a matrix"""
    cdef int format
    cdef cudssStatus_t status
    status = cudssMatrixGetFormat(<cudssMatrix_t> matrix, &format)
    check_status(status)
    return format

def get_build_version():
    return CUDSS_VERSION
