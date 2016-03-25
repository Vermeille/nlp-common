#include "gru.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

GRUParams::GRUParams(size_t output_size, size_t input_size)
    : wzx_(std::make_shared<Param>(output_size, input_size, Uniform(-0.08, 0.08))),
    wzh_(std::make_shared<Param>(output_size, output_size, Uniform(-0.08, 0.08))),
    bz_(std::make_shared<Param>(output_size, 1, Constant(0))),

    wrx_(std::make_shared<Param>(output_size, input_size, Uniform(-0.08, 0.08))),
    wrh_(std::make_shared<Param>(output_size, output_size, Uniform(-0.08, 0.08))),
    br_(std::make_shared<Param>(output_size, 1, Constant(0))),

    whx_(std::make_shared<Param>(output_size, input_size, Uniform(-0.08, 0.08))),
    whh_(std::make_shared<Param>(output_size, output_size, Uniform(-0.08, 0.08))),
    bh_(std::make_shared<Param>(output_size, 1, Constant(0))),

    hidden_(std::make_shared<Param>(output_size, 1, Gaussian(0, 0.1))) {
}

GRULayer::GRULayer(
        ComputationGraph& g, const GRUParams& params, bool learnable)
    : wzx_(g.CreateParam(params.wzx_, learnable)),
    wzh_(g.CreateParam(params.wzh_, learnable)),
    bz_(g.CreateParam(params.bz_, learnable)),

    wrx_(g.CreateParam(params.wrx_, learnable)),
    wrh_(g.CreateParam(params.wrh_, learnable)),
    br_(g.CreateParam(params.br_, learnable)),

    whx_(g.CreateParam(params.whx_, learnable)),
    whh_(g.CreateParam(params.whh_, learnable)),
    bh_(g.CreateParam(params.bh_, learnable)),

    hidden_(g.CreateParam(params.hidden_, learnable)) {
}

Var GRULayer::Step(Var x) {
    Var z = Sigmoid(wzx_ * x + wzh_ * hidden_ + bz_);
    Var r = Sigmoid(wrx_ * x + wrh_ * hidden_ + br_);
    Var h = Tanh(whx_ * x + (r ^ (whh_ * hidden_)) + bh_);

    return hidden_ = (z ^ hidden_) + ((1 - z) ^ h);
}

} // nn
} // ad
