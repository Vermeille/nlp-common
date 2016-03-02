#include "adadelta.h"

namespace ad {
namespace opt {

Eigen::MatrixXd& Adadelta::GetR(Var v) {
    auto r = r_.find(v.persistent_id());
    if (r == r_.end()) {
        r = r_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Eigen::MatrixXd(v.value().rows(), v.value().cols())
                )
            ).first;
        r->second.setZero();
    }
    return r->second;
}

Eigen::MatrixXd& Adadelta::GetS(Var v) {
    auto s = s_.find(v.persistent_id());
    if (s == s_.end()) {
        s = s_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Eigen::MatrixXd(v.value().rows(), v.value().cols())
                )
            ).first;
        s->second.setZero();
    }
    return s->second;
}


void Adadelta::Update(Var& v) {
    Eigen::MatrixXd& grad = v.derivative();
    Eigen::MatrixXd& r = GetR(v);
    Eigen::MatrixXd& s = GetS(v);

    Eigen::MatrixXd& param = v.value();
    for (size_t i = 0, nb_rows = param.rows(); i < nb_rows; ++i) {
        for (size_t j = 0, nb_cols = param.cols(); j < nb_cols; ++j) {
            double g = grad(i, j);
            r(i, j) = rho_ * r(i, j) + (1 - rho_) * g * g;
            double eta = learning_rate_ *
                (sqrt(s(i, j) + epsilon_) / sqrt(r(i, j) + epsilon_));
            s(i, j) = rho_ * s(i, j) + (1 - rho_) * (eta * g) * (eta * g);
            param(i, j) -= eta * g;
        }
    }
}

} // opt
} // ad
