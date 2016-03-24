#pragma once

#include <cublas_v2.h>
#include <memory>
#include <iostream>

#define CUDA_CALL(XXX) \
    do { auto err = XXX; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << \
        cudaGetErrorString(err) << ", at line " << __LINE__ \
        << std::endl; std::terminate(); } /*cudaDeviceSynchronize();*/} while (0)

namespace cuda {

class CublasHandle {
    cublasHandle_t cuhandle_;

    public:
        cublasHandle_t get() const { return cuhandle_; }

        CublasHandle() {
            cublasCreate(&cuhandle_);
        }

        ~CublasHandle() {
            cublasDestroy(cuhandle_);
        }
};

thread_local extern CublasHandle g_cuhandle;

} // cuda

namespace cuda {

// Strong typing for ptr allocated on the device
template <class T>
struct Ptr {
    T* ptr_;
    Ptr() : ptr_(nullptr) {}
    explicit Ptr(T* ptr) : ptr_(ptr) {}
    T* Get() const { return ptr_; }
    Ptr& operator=(const Ptr&) = default;
    Ptr At(size_t i) const { return Ptr(ptr_ + i); }
    bool operator==(Ptr a) const { return ptr_ == a.ptr_; }
    bool operator!=(Ptr a) const { return ptr_ != a.ptr_; }
    bool operator!=(std::nullptr_t) const { return ptr_; }
};

namespace helpers {

struct CUFree {
    template <class T>
    void operator()(T* ptr) const { CUDA_CALL(cudaFree(ptr)); }
};

template <class T>
using cunique_ptr = std::unique_ptr<T, CUFree>;

template <class T>
Ptr<T> cunew(size_t n) {
    T* ptr;
    CUDA_CALL(cudaMalloc((void**)&ptr, sizeof (T) * n));
    return Ptr<T>(ptr);
}

template <class T>
void CPUToGPU(Ptr<T> dst, const T* src, size_t n) {
    CUDA_CALL(cudaMemcpy(
            dst.Get(), src,
            sizeof(T) * n,
            cudaMemcpyHostToDevice));
}

template <class T>
void GPUToGPU(Ptr<T> dst, const Ptr<T> src, size_t n) {
    CUDA_CALL(cudaMemcpy(
            dst.Get(), src.Get(),
            sizeof(T) * n,
            cudaMemcpyDeviceToDevice));
}

template <class T>
void GPUToCPU(T* dst, const Ptr<T> src, size_t n) {
    CUDA_CALL(cudaMemcpy(
            dst, src.Get(),
            sizeof(T) * n,
            cudaMemcpyDeviceToHost));
}

void fill(Ptr<float> array, size_t size, float val);

} // helpers
} // cuda

namespace std {

template <class T>
struct hash<cuda::Ptr<T>> {
    size_t operator()(const cuda::Ptr<T>& s) const {
        return std::hash<T*>()(s.Get());
    }
};

} // std
