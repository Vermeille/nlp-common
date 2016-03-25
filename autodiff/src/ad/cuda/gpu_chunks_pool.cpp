#include "gpu_chunks_pool.h"

namespace cuda {

::cuda::Ptr<float> GPUChunksPool::Alloc(size_t sz) {
    auto& frees = frees_[sz];
    cuptr<float> res;
    if (frees.empty()) {
        res = ::cuda::helpers::cunew<float>(sz);
    } else {
        res = frees.back();
        frees.pop_back();
    }
    allocated_[res] = sz;
    return res;
}

void GPUChunksPool::Free(cuptr<float> ptr) {
    auto sz_iter = allocated_.find(ptr);
    if (sz_iter == allocated_.end()) {
        throw std::runtime_error("try to free a ptr not allocated");
    }
    size_t sz = sz_iter->second;
    allocated_.erase(sz_iter);
    frees_[sz].push_back(ptr);
}

thread_local GPUChunksPool g_gpu_pool;

} // cuda
