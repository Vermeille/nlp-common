#include "adam.h"

#include "../kernels/kernels.h"

namespace ad {
namespace opt {

Matrix& Adam::GetM(Var v) {
    auto m = m_.find(v.persistent_id());
    if (m == m_.end()) {
        m = m_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Matrix(v.value().rows(), v.value().cols())
                )
            ).first;
        m->second.SetZero();
    }
    return m->second;
}

Matrix& Adam::GetV(Var var) {
    auto v = v_.find(var.persistent_id());
    if (v == v_.end()) {
        v = v_.insert(
                std::make_pair(
                    var.persistent_id(),
                    Matrix(var.value().rows(), var.value().cols())
                )
            ).first;
        v->second.SetZero();
    }
    return v->second;
}


void Adam::Update(Var& var) {
    Matrix& param = var.value();
    ++t_;
    float fa = learning_rate_
        * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));

    auto g = cuda::Array(var.derivative().data());
    auto m = cuda::Array(GetM(var).data());
    auto v = cuda::Array(GetV(var).data());
    auto p = cuda::Array(param.data());
    auto a = cuda::Value(fa);
    auto beta1 = cuda::Value(beta1_);
    auto beta2 = cuda::Value(beta2_);
    auto one = cuda::Value(1.f);
    auto epsilon = cuda::Value(epsilon_);

    cuda::RunKernel(cuda::Seq(
            m = beta1 * m + (one - beta1) * g,
            v = beta2 * v + (one - beta2) * g * g,
            p -= a * m / (cuda::sqrt(v) + epsilon)),
        param.size());
}

} // opt
} // ad
