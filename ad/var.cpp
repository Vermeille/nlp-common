#include "var.h"

namespace ad {

void DoNothing(Val&) {}

Val::Val(size_t rows,
        size_t cols,
        backprop_t bp,
        std::shared_ptr<Val> dep1,
        std::shared_ptr<Val> dep2) :
    a_(rows, cols),
    da_(Eigen::MatrixXd::Zero(rows, cols)),
    dep1_(dep1),
    dep2_(dep2),
    backprop_(bp),
    learnable_(false) {
}

Val::Val(Eigen::MatrixXd a,
        backprop_t bp,
        std::shared_ptr<Val> dep1,
        std::shared_ptr<Val> dep2) :
    a_(a),
    da_(Eigen::MatrixXd::Zero(a.rows(), a.cols())),
    dep1_(dep1),
    dep2_(dep2),
    backprop_(bp),
    learnable_(false) {
}

void Val::Backprop() {
    backprop_(*this);
    if (dep1_)
        dep1_->Backprop();
    if (dep2_)
        dep2_->Backprop();
}

void Val::ClearGrad() {
    da_.setZero();
    if (dep1_)
        dep1_->ClearGrad();
    if (dep2_)
        dep2_->ClearGrad();
}

void Val::Resize(size_t rows, size_t cols) {
    a_.conservativeResize(rows, cols);
    da_.resize(rows, cols);
}

}
