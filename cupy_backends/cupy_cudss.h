#ifndef INCLUDE_GUARD_CUPY_CUDSS_H
#define INCLUDE_GUARD_CUPY_CUDSS_H

#ifdef CUPY_USE_HIP

#include "stub/cupy_cudss.h"

#elif !defined(CUPY_NO_CUDA)

#include <cudss.h>

#else

#include "stub/cupy_cudss.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUDSS_H
