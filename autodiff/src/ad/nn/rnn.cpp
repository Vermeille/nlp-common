#include "rnn.h"

#include <stdexcept>

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

RNNLayerParams::RNNLayerParams(int out_sz, int in_sz) :
        whx_(std::make_shared<Param>(out_sz, in_sz, Xavier())),
        whh_(std::make_shared<Param>(out_sz, out_sz, Xavier())),
        bh_(std::make_shared<Param>(out_sz, 1, Constant(0))),
        h_(std::make_shared<Param>(out_sz, 1, Gaussian(0, 0.1))) {
}

RNNLayer::RNNLayer(
            ComputationGraph& g,
            const RNNLayerParams& params,
            bool learnable) :
        whx_(g.CreateParam(params.whx_, learnable)),
        whh_(g.CreateParam(params.whh_, learnable)),
        bh_(g.CreateParam(params.bh_, learnable)),
        h_(g.CreateParam(params.h_, learnable)) {
}

Var RNNLayer::Step(Var x) {
    return h_ = Sigmoid(whx_ * x + whh_ * h_ + bh_);
}

void RNNLayerParams::Serialize(std::ostream& out) const {
    out << "RNN1\n";
    utils::WriteMatrixTxt(whx_->value(), out);
    utils::WriteMatrixTxt(whh_->value(), out);
    utils::WriteMatrixTxt(bh_->value(), out);
}

RNNLayerParams RNNLayerParams::FromSerialized(std::istream& in) {
    std::string magic;
    in >> magic;
    if (magic != "RNN") {
        throw std::runtime_error("Not a RNN1 layer, but " + magic);
    }

    RNNLayerParams rnn(0, 0);
    rnn.whx_->value() = utils::ReadMatrixTxt(in);
    rnn.whh_->value() = utils::ReadMatrixTxt(in);
    rnn.bh_->value() = utils::ReadMatrixTxt(in);
    return rnn;
}

} // nn
} // ad
