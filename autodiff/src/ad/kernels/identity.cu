#include "kernels.h"

namespace ad {
namespace cuda {

__global__ void cuSetIdentity(float* array, size_t rows, size_t cols) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    for (; x < rows; x += blockDim.x * gridDim.x) {
        for (; y < cols; y += blockDim.y * gridDim.y) {
            array[y * rows + x] = (x == y) ? 1 : 0;
        }
    }
}

void SetIdentity(float* array, size_t rows, size_t cols) {
    dim3 block(16, 16);
    dim3 grid((rows + 16 - 1) / 16, (cols + 16 - 1) / 16);
    cuSetIdentity<<<grid, block>>>(array, rows, cols);
    cudaDeviceSynchronize();
}

} // cuda
} // ad
