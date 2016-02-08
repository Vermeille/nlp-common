#include "lstm.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

LSTMLayer::LSTMLayer(size_t output_size, size_t input_size)
    : wix_(std::make_shared<Eigen::MatrixXd>(output_size, input_size)),
    wih_(std::make_shared<Eigen::MatrixXd>(output_size, output_size)),
    bi_(std::make_shared<Eigen::MatrixXd>(output_size, 1)),

    wfx_(std::make_shared<Eigen::MatrixXd>(output_size, input_size)),
    wfh_(std::make_shared<Eigen::MatrixXd>(output_size, output_size)),
    bf_(std::make_shared<Eigen::MatrixXd>(output_size, 1)),

    wox_(std::make_shared<Eigen::MatrixXd>(output_size, input_size)),
    woh_(std::make_shared<Eigen::MatrixXd>(output_size, output_size)),
    bo_(std::make_shared<Eigen::MatrixXd>(output_size, 1)),

    wcx_(std::make_shared<Eigen::MatrixXd>(output_size, input_size)),
    wch_(std::make_shared<Eigen::MatrixXd>(output_size, output_size)),
    bc_(std::make_shared<Eigen::MatrixXd>(output_size, 1)) {
        utils::RandomInit(*wix_, -1, 1);
        utils::RandomInit(*wih_, -1, 1);
        utils::RandomInit(*bi_, -1, 1);

        utils::RandomInit(*wfx_, -1, 1);
        utils::RandomInit(*wfh_, -1, 1);
        utils::RandomInit(*bf_, -1, 1);

        utils::RandomInit(*wox_, -1, 1);
        utils::RandomInit(*woh_, -1, 1);
        utils::RandomInit(*bo_, -1, 1);

        utils::RandomInit(*wcx_, -1, 1);
        utils::RandomInit(*wch_, -1, 1);
        utils::RandomInit(*bc_, -1, 1);
}


NeuralOutput<std::pair<std::vector<Var>, Var>> LSTMLayer::ComputeWithHidden(
        const NeuralOutput<std::vector<Var>>& in) const {
    std::vector<Var> out;
    out.reserve(in.out.size());

    ComputationGraph* g = in.out[0].graph();

    Var wix = g->CreateParam(wix_);
    Var wih = g->CreateParam(wih_);
    Var bi = g->CreateParam(bi_);

    Var wfx = g->CreateParam(wfx_);
    Var wfh = g->CreateParam(wfh_);
    Var bf = g->CreateParam(bf_);

    Var wox = g->CreateParam(wox_);
    Var woh = g->CreateParam(woh_);
    Var bo = g->CreateParam(bo_);

    Var wcx = g->CreateParam(wcx_);
    Var wch = g->CreateParam(wch_);
    Var bc = g->CreateParam(bc_);

    Eigen::MatrixXd zero(wox_->rows(), 1);
    zero.setZero();

    Var hidden = g->CreateParam(zero);
    Var cell_prev = g->CreateParam(zero);

    for (Var x : in.out) {
        Var input_gate = Sigmoid(wix * x + wih * hidden + bi);
        Var forget_gate = Sigmoid(wfx * x + wfh * hidden + bf);
        Var output_gate = Sigmoid(wox * x + woh * hidden + bo);
        Var cell_write = Tanh(wcx * x + wch * hidden + bc);

        Var cell_d = (forget_gate ^ cell_prev) + (input_gate ^ cell_write);

        Var hidden_d = output_gate ^ Tanh(cell_d);

        hidden = hidden_d;
        cell_prev = cell_d;
        out.push_back(hidden);
    }

    return in.Forward(std::make_pair(out, cell_prev), {
            wix, wih, bi,
            wfx, wfh, bf,
            wox, woh, bo,
            wcx, wch, bc });
}

void LSTMLayer::ResizeInput(size_t in) {
    utils::RandomExpandMatrix(*wix_, wix_->rows(), in, -1, 1);
    utils::RandomExpandMatrix(*wfx_, wix_->rows(), in, -1, 1);
    utils::RandomExpandMatrix(*wox_, wix_->rows(), in, -1, 1);
    utils::RandomExpandMatrix(*wcx_, wix_->rows(), in, -1, 1);
}

void LSTMLayer::ResizeOutput(size_t out) {
    size_t hidden_size = wix_->cols();
    utils::RandomExpandMatrix(*wix_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*wih_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*bi_, out, 1, -1, 1);

    utils::RandomExpandMatrix(*wfx_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*wfh_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*bf_, out, 1, -1, 1);

    utils::RandomExpandMatrix(*wox_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*woh_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*bo_, out, 1, -1, 1);

    utils::RandomExpandMatrix(*wcx_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*wch_, out, hidden_size, -1, 1);
    utils::RandomExpandMatrix(*bc_, out, 1, -1, 1);
}

} // nn
} // ad
