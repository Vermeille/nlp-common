#pragma once

#include <vector>
#include <unordered_map>
#include "helpers.h"

namespace cuda {

class GPUChunksPool {
    public:
        template <class T>
        using cuptr = ::cuda::Ptr<T>;

    private:
        std::unordered_map<int, std::vector<cuptr<float>>> frees_;
        std::unordered_map<cuptr<float>, size_t> allocated_;

    public:
        ::cuda::Ptr<float> Alloc(size_t sz);
        void Free(cuptr<float> ptr);
};

extern thread_local GPUChunksPool g_gpu_pool;

} // cuda
