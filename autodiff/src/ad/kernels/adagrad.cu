#include "kernels.h"

namespace ad {
namespace cuda {

__global__
void cuAdagrad(
        float* p,
        float* sqgrad,
        float* g,
        float learning_rate,
        float epsilon,
        size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        sqgrad[i] += g[i] * g[i];
        p[i] -= (learning_rate / sqrt(sqgrad[i] + epsilon)) * g[i];
    }
}

void Adagrad(
        float* p,
        float* sqgrad,
        float* g,
        float learning_rate,
        float epsilon,
        size_t sz) {
    cuAdagrad<<<(sz + 128 - 1) / 128, 128>>>(
        p, sqgrad, g, learning_rate, epsilon, sz);
    cudaDeviceSynchronize();
}

} // cuda
} // ad
