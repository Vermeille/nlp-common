#include "adagrad.h"

namespace ad {
namespace opt {

Eigen::MatrixXd& Adagrad::AccumulatedGradient(Var v) {
    auto sqgrad = grad_acc_.find(v.id());
    if (sqgrad == grad_acc_.end()) {
        sqgrad = grad_acc_.insert(
                std::make_pair(
                    v.id(),
                    Eigen::MatrixXd(v.value().rows(), v.value().cols())
                )
            ).first;
        sqgrad->second.setZero();
    }
    return sqgrad->second;
}


void Adagrad::Update(Var& v) {
    if (v.derivative().isZero()) {
        return;
    }

    Eigen::MatrixXd cur_grad = v.derivative();
    Eigen::MatrixXd& sqgrad = AccumulatedGradient(v);
    sqgrad += cur_grad.cwiseProduct(cur_grad);

    Eigen::MatrixXd& param = v.value();
    for (size_t i = 0, nb_rows = param.rows(); i < nb_rows; ++i) {
        for (size_t j = 0, nb_cols = param.cols(); j < nb_cols; ++j) {
            param(i, j) -= (learning_rate_ / sqrt(sqgrad(i, j) + epsilon_))
                * cur_grad(i, j);
        }
    }
}

} // opt
} // ad
