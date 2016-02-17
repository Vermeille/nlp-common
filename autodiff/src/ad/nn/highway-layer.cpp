#include "highway-layer.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

HighwayLayerParams::HighwayLayerParams(size_t sz, double init)
        : w_(std::make_shared<Param>(sz, sz, init)),
        wt_(std::make_shared<Param>(sz, sz, init)),
        wc_(std::make_shared<Param>(sz, sz, init)) {
}

void HighwayLayerParams::Resize(size_t sz, double init) {
    utils::RandomExpandMatrix(w_->value(), sz, sz, -init, init);
    utils::RandomExpandMatrix(wt_->value(), sz, sz, -init, init);
    utils::RandomExpandMatrix(wc_->value(), sz, sz, -init, init);
}

HighwayLayer::HighwayLayer(
        ComputationGraph& g, const HighwayLayerParams& params, bool learnable)
        : w_(g.CreateParam(params.w_, learnable)),
        wt_(g.CreateParam(params.wt_, learnable)),
        wc_(g.CreateParam(params.wc_, learnable)) {
}

Var HighwayLayer::Step(Var x) const {
    return (Relu(w_ * x) ^ Sigmoid(wt_ * x)) + (x ^ Sigmoid(wc_ * x));
}

} // nn
} // ad
