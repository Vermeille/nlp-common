#include "fullyconn.h"

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

FullyConnLayer::FullyConnLayer(int out_sz, int in_sz) :
    w_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
    b_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)) {
        ad::utils::RandomInit(*w_ , -1, 1);
        ad::utils::RandomInit(*b_ , -1, 1);
    }

NeuralOutput<Var> FullyConnLayer::Compute(NeuralOutput<Var> in) {
    Var w = in.out.graph()->CreateParam(w_);
    Var b = in.out.graph()->CreateParam(b_);

    return in.Forward(w * in.out + b, {w, b});
}

void FullyConnLayer::ResizeOutput(int size) {
    utils::RandomExpandMatrix(*w_, size, w_->cols(), -1, 1);
    utils::RandomExpandMatrix(*b_, size, 1, -1, 1);
}

void FullyConnLayer::ResizeInput(int size) {
    utils::RandomExpandMatrix(*w_, w_->rows(), size, -1, 1);
}

} // nn
} // ad
