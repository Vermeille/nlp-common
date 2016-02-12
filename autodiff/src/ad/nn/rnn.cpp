#include "rnn.h"

#include <stdexcept>

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

RNNLayerParams::RNNLayerParams(int out_sz, int in_sz, double init) :
        whx_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
        whh_(std::make_shared<Eigen::MatrixXd>(out_sz, out_sz)),
        bh_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        h_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)) {
    ad::utils::RandomInit(*whx_ , -init, init);
    ad::utils::RandomInit(*whh_ , -init, init);
    ad::utils::RandomInit(*bh_ , -init, init);
    ad::utils::RandomInit(*h_ , -init, init);
}

RNNLayer::RNNLayer(ComputationGraph& g, const RNNLayerParams& params) :
        whx_(g.CreateParam(params.whx_)),
        whh_(g.CreateParam(params.whh_)),
        bh_(g.CreateParam(params.bh_)),
        h_(g.CreateParam(params.h_)) {
}

Var RNNLayer::Step(Var x) {
    return h_ = Sigmoid(whx_ * x + whh_ * h_ + bh_);
}

void RNNLayerParams::Serialize(std::ostream& out) const {
    out << "RNN1\n";
    utils::WriteMatrixTxt(*whx_, out);
    utils::WriteMatrixTxt(*whh_, out);
    utils::WriteMatrixTxt(*bh_, out);
}

RNNLayerParams RNNLayerParams::FromSerialized(std::istream& in) {
    std::string magic;
    in >> magic;
    if (magic != "RNN") {
        throw std::runtime_error("Not a RNN1 layer, but " + magic);
    }

    RNNLayerParams rnn(0, 0);
    *rnn.whx_ = utils::ReadMatrixTxt(in);
    *rnn.whh_ = utils::ReadMatrixTxt(in);
    *rnn.bh_ = utils::ReadMatrixTxt(in);
    return rnn;
}

} // nn
} // ad
