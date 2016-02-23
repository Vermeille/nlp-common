#include "fullyconn.h"

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

FullyConnParams::FullyConnParams(int out_sz, int in_sz) :
    w_(std::make_shared<Param>(out_sz, in_sz, Xavier())),
    b_(std::make_shared<Param>(out_sz, 1, Constant(0))) {
}

FullyConnLayer::FullyConnLayer(
        ComputationGraph& g, const FullyConnParams& params, bool learnable) :
    w_(g.CreateParam(params.w_, learnable)),
    b_(g.CreateParam(params.b_, learnable)) {
}

Var FullyConnLayer::Compute(Var x) const {
    return w_ * x + b_;
}

void FullyConnParams::ResizeOutput(int size, double init) {
    utils::RandomExpandMatrix(w_->value(), size, w_->cols(), -init, init);
    utils::RandomExpandMatrix(b_->value(), size, 1, -init, init);
}

void FullyConnParams::ResizeInput(int size, double init) {
    utils::RandomExpandMatrix(w_->value(), w_->rows(), size, -init, init);
}

void FullyConnParams::Serialize(std::ostream& out) const {
    out << "FULLY-CONN\n";
    utils::WriteMatrixTxt(w_->value(), out);
    utils::WriteMatrixTxt(b_->value(), out);
}

FullyConnParams FullyConnParams::FromSerialized(std::istream& in) {
    std::string magic;
    in >> magic;

    if (magic != "FULLY-CONN") {
        throw std::runtime_error("This is not a fullyconn layer, '"
                + magic + "' found instead");
    }

    FullyConnParams fc(0, 0);
    fc.w_ = std::make_shared<Param>(utils::ReadMatrixTxt(in));
    fc.b_ = std::make_shared<Param>(utils::ReadMatrixTxt(in));
    return fc;
}

std::vector<Var> FullyConnLayer::Params() const {
    return { w_, b_ };
}

} // nn
} // ad
