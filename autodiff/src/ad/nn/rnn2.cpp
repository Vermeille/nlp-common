#include "rnn2.h"

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

RNNLayer2::RNNLayer2(int out_sz, int in_sz, int hidden_sz) :
        whx_(std::make_shared<Eigen::MatrixXd>(hidden_sz, in_sz)),
        whh_(std::make_shared<Eigen::MatrixXd>(hidden_sz, hidden_sz)),
        wox_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
        woh_(std::make_shared<Eigen::MatrixXd>(out_sz, hidden_sz)),
        bo_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        bh_(std::make_shared<Eigen::MatrixXd>(hidden_sz, 1)) {
    ad::utils::RandomInit(*whx_ , -1, 1);
    ad::utils::RandomInit(*whh_ , -1, 1);
    ad::utils::RandomInit(*wox_ , -1, 1);
    ad::utils::RandomInit(*woh_ , -1, 1);
    ad::utils::RandomInit(*bo_ , -1, 1);
    ad::utils::RandomInit(*bh_ , -1, 1);
}

NeuralOutput<std::pair<std::vector<Var>, Var>> RNNLayer2::Compute(
        NeuralOutput<std::vector<Var>> in) const {
    ComputationGraph* g = in.out[0].graph();
    Var whx = g->CreateParam(whx_);
    Var whh = g->CreateParam(whh_);
    Var wox = g->CreateParam(wox_);
    Var woh = g->CreateParam(woh_);
    Var bo = g->CreateParam(bo_);
    Var bh = g->CreateParam(bh_);

    Eigen::MatrixXd zero(whx.value().rows(), 1);
    zero.setZero();
    Var h = g->CreateParam(zero);

    std::vector<Var> out;

    for (Var x : in.out) {
        out.push_back(wox * x + woh * h + bo);
        h = Sigmoid(whx * x + whh * h + bh);
    }
    return in.Forward(
            std::make_pair(out, h), {whx, whh, wox, woh, bo, bh});
}

void RNNLayer2::Serialize(std::ostream& out) const {
    out << "RNN2\n";
    utils::WriteMatrixTxt(*whx_, out);
    utils::WriteMatrixTxt(*whh_, out);
    utils::WriteMatrixTxt(*bh_, out);
    utils::WriteMatrixTxt(*wox_, out);
    utils::WriteMatrixTxt(*woh_, out);
    utils::WriteMatrixTxt(*bo_, out);
}

RNNLayer2 RNNLayer2::FromSerialized(std::istream& in) {
    std::string magic;
    in >> magic;
    if (magic != "RNN2") {
        throw std::runtime_error("Not a RNN2 layer, but " + magic);
    }

    RNNLayer2 rnn(0, 0, 0);
    *rnn.whx_ = utils::ReadMatrixTxt(in);
    *rnn.whh_ = utils::ReadMatrixTxt(in);
    *rnn.bh_ = utils::ReadMatrixTxt(in);
    *rnn.wox_ = utils::ReadMatrixTxt(in);
    *rnn.woh_ = utils::ReadMatrixTxt(in);
    *rnn.bo_ = utils::ReadMatrixTxt(in);
    return rnn;
}

} // nn
} // ad
