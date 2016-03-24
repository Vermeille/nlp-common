#include "kernels.h"

namespace ad {
namespace cuda {

__global__
void cuLog(float* res, const float* arr1, size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        res[i] = log(arr1[i]);
    }
}

void Log(float* res, const float* arr1, size_t sz) {
    cuLog<<<(sz + 128 - 1) / 128, 128>>>(res, arr1, sz);
    cudaDeviceSynchronize();
}

__global__
void cuLogGrad(float* da, const float* dx, const float* a, size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        da[i] += dx[i] / a[i];
    }
}

void LogGrad(float* da, const float* dx, const float* a, size_t sz) {
    cuLogGrad<<<(sz + 128 - 1) / 128, 128>>>(da, dx, a, sz);
    cudaDeviceSynchronize();
}

} // cuda
} // ad
