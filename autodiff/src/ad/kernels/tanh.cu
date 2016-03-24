#include "kernels.h"

namespace ad {
namespace cuda {

__global__
void cuTanh(float* res, const float* arr1, size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        res[i] = tanhf(arr1[i]);
    }
}

void Tanh(float* res, const float* arr1, size_t sz) {
    cuTanh<<<(sz + 128 - 1) / 128, 128>>>(res, arr1, sz);
    cudaDeviceSynchronize();
}

__global__
void cuTanhGrad(float* res, const float* in, size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        float denom = coshf(2 * in[i]) + 1;
        denom = denom * denom;
        float num = coshf(in[i]);
        num = 4 * num * num;
        res[i] = num / denom;
    }
}

void TanhGrad(float* res, const float* arr1, size_t sz) {
    cuTanhGrad<<<(sz + 128 - 1) / 128, 128>>>(res, arr1, sz);
    cudaDeviceSynchronize();
}

} // cuda
} // ad
