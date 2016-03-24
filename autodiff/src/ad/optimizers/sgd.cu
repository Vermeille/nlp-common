#include "sgd.h"

#include "../kernels/kernels.h"
#include "../graph.h"
#include "../matrix.h"

namespace ad {
namespace opt {

void SGD::Update(ad::Var& v) {
    auto alpha = cuda::Value(alpha_);
    auto g = cuda::Array(v.derivative().data());
    auto p = cuda::Array(v.value().data());

    cuda::RunKernel(
            p -= alpha * g,
        v.value().size());
}

} // opt
} // ad
