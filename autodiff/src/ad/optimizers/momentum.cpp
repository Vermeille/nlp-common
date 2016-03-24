#include "momentum.h"

namespace ad {
namespace opt {

Matrix& Momentum::AccumulatedVelocity(Var v) {
    auto vel = velocity_.find(v.persistent_id());
    if (vel == velocity_.end()) {
        vel = velocity_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Matrix(v.value().rows(), v.value().cols())
                )
            ).first;
        vel->second.SetZero();
    }
    return vel->second;
}


void Momentum::Update(Var& v) {
    Matrix& cur_grad = v.derivative();
    Matrix& vel = AccumulatedVelocity(v);
    vel = (momentum_ * vel) - learning_rate_ * cur_grad;

    v.value() += vel;
}

} // opt
} // ad
