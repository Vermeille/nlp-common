#include "adadelta.h"
#include "../kernels/kernels.h"

namespace ad {
namespace opt {

Matrix& Adadelta::GetR(Var v) {
    auto r = r_.find(v.persistent_id());
    if (r == r_.end()) {
        r = r_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Matrix(v.value().rows(), v.value().cols())
                )
            ).first;
        r->second.SetZero();
    }
    return r->second;
}

Matrix& Adadelta::GetS(Var v) {
    auto s = s_.find(v.persistent_id());
    if (s == s_.end()) {
        s = s_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Matrix(v.value().rows(), v.value().cols())
                )
            ).first;
        s->second.SetZero();
    }
    return s->second;
}


struct AdadeltaKernel {
    float* p_;
    float* r_;
    float* s_;
    float* g_;
    float learning_rate_;
    float rho_;
    float epsilon_;

    __device__
    inline
    void operator()(size_t i) {
        r_[i] = rho_ * r_[i] + (1 - rho_) * g_[i] * g_[i];
        float eta = learning_rate_
            * (sqrt(s_[i] + epsilon_) / sqrtf(r_[i] + epsilon_));
        s_[i] = rho_ * s_[i] + (1 - rho_) * (eta * g_[i]) * (eta * g_[i]);
        p_[i] -= eta * g_[i];
    }
};

void Adadelta::Update(Var& v) {
    Matrix& param = v.value();

    AdadeltaKernel kern {
        param.data().Get(),
        GetR(v).data().Get(),
        GetS(v).data().Get(),
        v.derivative().data().Get(),
        learning_rate_,
        rho_,
        epsilon_
    };

    cuda::RunKernel(kern, param.size());
}

} // opt
} // ad
