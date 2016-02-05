#include "rnn-encoder.h"

#include "../graph.h"
#include "../operators.h"

namespace ad {
namespace nn {

RNNEncoderLayer::RNNEncoderLayer(int out_sz, int in_sz) :
        whx_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
        whh_(std::make_shared<Eigen::MatrixXd>(out_sz, out_sz)),
        bh_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)) {
    ad::utils::RandomInit(*whx_ , -1, 1);
    ad::utils::RandomInit(*whh_ , -1, 1);
    ad::utils::RandomInit(*bh_ , -1, 1);
}

NeuralOutput<Var> RNNEncoderLayer::Compute(NeuralOutput<std::vector<Var>> in) const {
    ComputationGraph* g = in.out[0].graph();
    Var whx = g->CreateParam(whx_);
    Var whh = g->CreateParam(whh_);
    Var bh = g->CreateParam(bh_);

    Eigen::MatrixXd zero(whx.value().rows(), 1);
    zero.setZero();
    Var h = g->CreateParam(zero);

    for (Var x : in.out) {
        h = Sigmoid(whx * x + whh * h + bh);
    }

    return in.Forward(h, {whx, whh, bh});
}

} // nn
} // ad
