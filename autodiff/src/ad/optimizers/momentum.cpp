#include "momentum.h"

namespace ad {
namespace opt {

Eigen::MatrixXd& Momentum::AccumulatedVelocity(Var v) {
    auto vel = velocity_.find(v.persistent_id());
    if (vel == velocity_.end()) {
        vel = velocity_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Eigen::MatrixXd(v.value().rows(), v.value().cols())
                )
            ).first;
        vel->second.setZero();
    }
    return vel->second;
}


void Momentum::Update(Var& v) {
    Eigen::MatrixXd& cur_grad = v.derivative();
    Eigen::MatrixXd& vel = AccumulatedVelocity(v);
    vel = (momentum_ * vel.array()) - learning_rate_ * cur_grad.array();

    v.value() += vel;
}

} // opt
} // ad
