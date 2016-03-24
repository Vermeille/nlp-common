#include "kernels.h"

namespace ad {
namespace cuda {

__global__
void cuAdam(
        float* p,
        float* m,
        float* v,
        float* g,
        float beta1,
        float beta2,
        float a,
        float epsilon,
        size_t sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        p[i] -= a * m[i] / (sqrt(v[i]) + epsilon);
    }
}

void Adam(
        float* p,
        float* m,
        float* v,
        float* g,
        float beta1,
        float beta2,
        float a,
        float epsilon,
        size_t sz) {
    cuAdam<<<(sz + 128 - 1) / 128, 128>>>(
            p, m, v, g, beta1, beta2, a, epsilon, sz);
    cudaDeviceSynchronize();
}

} // cuda
} // ad
