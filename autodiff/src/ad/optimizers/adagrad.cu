#include "adagrad.h"

#include "../kernels/kernels.h"

namespace ad {
namespace opt {

Matrix& Adagrad::AccumulatedGradient(Var v) {
    auto sqgrad = grad_acc_.find(v.persistent_id());
    if (sqgrad == grad_acc_.end()) {
        sqgrad = grad_acc_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Matrix(v.value().rows(), v.value().cols())
                )
            ).first;
        sqgrad->second.SetZero();
    }
    return sqgrad->second;
}


void Adagrad::Update(Var& v) {
    Matrix& param = v.value();
    Matrix& cur_grad = v.derivative();
    Matrix& sqgrad = AccumulatedGradient(v);

    auto p = cuda::Array(param.data());
    auto sq = cuda::Array(sqgrad.data());
    auto g = cuda::Array(cur_grad.data());
    auto learning_rate = cuda::Value(learning_rate_);
    auto epsilon = cuda::Value(epsilon_);

    cuda::RunKernel(cuda::Seq(
            sq += g * g,
            p -= (learning_rate / cuda::sqrt(sq + epsilon)) * g),
        param.size());
}

} // opt
} // ad
