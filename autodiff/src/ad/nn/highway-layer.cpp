#include "highway-layer.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

HighwayLayerParams::HighwayLayerParams(size_t sz, double init)
        : w_(std::make_shared<Eigen::MatrixXd>(sz, sz)),
        wt_(std::make_shared<Eigen::MatrixXd>(sz, sz)),
        wc_(std::make_shared<Eigen::MatrixXd>(sz, sz)) {
    ad::utils::RandomInit(*w_ , -init, init);
    ad::utils::RandomInit(*wt_ , -init, init);
    ad::utils::RandomInit(*wc_ , -init, init);
}

void HighwayLayerParams::Resize(size_t sz, double init) {
    utils::RandomExpandMatrix(*w_, sz, sz, -init, init);
    utils::RandomExpandMatrix(*wt_, sz, sz, -init, init);
    utils::RandomExpandMatrix(*wc_, sz, sz, -init, init);
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
