#include "helpers.h"

namespace cuda {

thread_local CublasHandle g_cuhandle;

namespace helpers {

__global__ void cuFill(float* array, size_t size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        array[i] = val;
    }
}

void fill(Ptr<float> array, size_t size, float val) {
    cuFill<<<(size + 512 - 1) / 512, 512>>>(array.Get(), size, val);
}

} // helpers
} // cuda


