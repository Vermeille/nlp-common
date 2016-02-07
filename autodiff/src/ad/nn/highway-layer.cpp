#include "highway-layer.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

HighwayLayer::HighwayLayer(size_t out, size_t in)
     : w_(std::make_shared<Eigen::MatrixXd>(out, in)),
     wt_(std::make_shared<Eigen::MatrixXd>(out, in)),
     wc_(std::make_shared<Eigen::MatrixXd>(out, in)) {
}

NeuralOutput<Var> HighwayLayer::Compute(NeuralOutput<Var> in) const {
    ComputationGraph* g = in.out.graph();
    Var w = g->CreateParam(w_);
    Var wt = g->CreateParam(wt_);
    Var wc = g->CreateParam(wc_);
    return in.Forward(
            (Relu(w * in.out) ^ Sigmoid(wt * in.out))
            + (in.out ^ Sigmoid(wc * in.out)), {w, wt, wc});
}

void HighwayLayer::ResizeInput(size_t in) {
    utils::RandomExpandMatrix(*w_, w_->rows(), in, -1, 1);
    utils::RandomExpandMatrix(*wt_, w_->rows(), in, -1, 1);
    utils::RandomExpandMatrix(*wc_, w_->rows(), in, -1, 1);
}

void HighwayLayer::ResizeOutput(size_t in) {
    utils::RandomExpandMatrix(*w_, in, w_->cols(), -1, 1);
    utils::RandomExpandMatrix(*wt_, in, w_->cols(), -1, 1);
    utils::RandomExpandMatrix(*wc_, in, w_->cols(), -1, 1);
}

} // nn
} // ad
