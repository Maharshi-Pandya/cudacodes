import ctypes

CUBLAS_STATUS_SUCCESS = 0

# load the cublas shared library
cublas = ctypes.cdll.LoadLibrary("libcublas.so")

cublas.cublasCreate_v2.restype = ctypes.c_int
cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

cublas.cublasDestroy_v2.restype = ctypes.c_int
cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]


def cublas_create_handle():
    handle = ctypes.c_void_p()
    status = cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cublasCreate failed with status {status}")
    
    return handle

def cublas_destroy_handle(handle):
    status = cublas.cublasDestroy_v2(handle)
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cublasDestroy failed with status {status}")
