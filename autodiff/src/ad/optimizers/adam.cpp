#include "adam.h"

namespace ad {
namespace opt {

Eigen::MatrixXd& Adam::GetM(Var v) {
    auto m = m_.find(v.persistent_id());
    if (m == m_.end()) {
        m = m_.insert(
                std::make_pair(
                    v.persistent_id(),
                    Eigen::MatrixXd(v.value().rows(), v.value().cols())
                )
            ).first;
        m->second.setZero();
    }
    return m->second;
}

Eigen::MatrixXd& Adam::GetV(Var var) {
    auto v = v_.find(var.persistent_id());
    if (v == v_.end()) {
        v = v_.insert(
                std::make_pair(
                    var.persistent_id(),
                    Eigen::MatrixXd(var.value().rows(), var.value().cols())
                )
            ).first;
        v->second.setZero();
    }
    return v->second;
}


void Adam::Update(Var& var) {
    Eigen::MatrixXd& grad = var.derivative();
    Eigen::MatrixXd& m = GetM(var);
    Eigen::MatrixXd& v = GetV(var);
    Eigen::MatrixXd& param = var.value();

    ++t_;
    double a = learning_rate_
        * sqrt(1 - pow(beta2_, t_)) / (1 - pow(beta1_, t_));
    for (size_t i = 0, nb_rows = param.rows(); i < nb_rows; ++i) {
        for (size_t j = 0, nb_cols = param.cols(); j < nb_cols; ++j) {
            double g = grad(i, j);
            m(i, j) = beta1_ * m(i, j) + (1 - beta1_) * g;
            v(i, j) = beta2_ * v(i, j) + (1 - beta2_) * g * g;
            param(i, j) -= a * m(i, j) / (sqrt(v(i, j)) + epsilon_);
        }
    }
}

} // opt
} // ad
