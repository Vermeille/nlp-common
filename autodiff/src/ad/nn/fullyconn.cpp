#include "fullyconn.h"

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

FullyConnParams::FullyConnParams(int out_sz, int in_sz, double init) :
    w_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
    b_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)) {
        ad::utils::RandomInit(*w_ , -init, init);
        ad::utils::RandomInit(*b_ , -init, init);
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
    utils::RandomExpandMatrix(*w_, size, w_->cols(), -init, init);
    utils::RandomExpandMatrix(*b_, size, 1, -init, init);
}

void FullyConnParams::ResizeInput(int size, double init) {
    utils::RandomExpandMatrix(*w_, w_->rows(), size, -init, init);
}

void FullyConnParams::Serialize(std::ostream& out) const {
    out << "FULLY-CONN\n";
    utils::WriteMatrixTxt(*w_, out);
    utils::WriteMatrixTxt(*b_, out);
}

FullyConnParams FullyConnParams::FromSerialized(std::istream& in) {
    std::string magic;
    in >> magic;

    if (magic != "FULLY-CONN") {
        throw std::runtime_error("This is not a fullyconn layer, '"
                + magic + "' found instead");
    }

    FullyConnParams fc(0, 0);
    fc.w_ = std::make_shared<Eigen::MatrixXd>(utils::ReadMatrixTxt(in));
    fc.b_ = std::make_shared<Eigen::MatrixXd>(utils::ReadMatrixTxt(in));
    return fc;
}

std::vector<Var> FullyConnLayer::Params() const {
    return { w_, b_ };
}

} // nn
} // ad
