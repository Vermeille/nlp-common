#include "lstm.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

LSTMParams::LSTMParams(size_t output_size, size_t input_size, double init)
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
    bc_(std::make_shared<Eigen::MatrixXd>(output_size, 1)),

    cell_(std::make_shared<Eigen::MatrixXd>(output_size, 1)),
    hidden_(std::make_shared<Eigen::MatrixXd>(output_size, 1)) {
        utils::RandomInit(*wix_, -init, init);
        utils::RandomInit(*wih_, -init, init);
        utils::RandomInit(*bi_, -init, init);

        utils::RandomInit(*wfx_, -init, init);
        utils::RandomInit(*wfh_, -init, init);
        utils::RandomInit(*bf_, -init, init);

        utils::RandomInit(*wox_, -init, init);
        utils::RandomInit(*woh_, -init, init);
        utils::RandomInit(*bo_, -init, init);

        utils::RandomInit(*wcx_, -init, init);
        utils::RandomInit(*wch_, -init, init);
        utils::RandomInit(*bc_, -init, init);

        utils::RandomInit(*cell_, -init, init);
        utils::RandomInit(*hidden_, -init, init);
}


void LSTMParams::ResizeInput(size_t in, double init) {
    utils::RandomExpandMatrix(*wix_, wix_->rows(), in, -init, init);
    utils::RandomExpandMatrix(*wfx_, wfx_->rows(), in, -init, init);
    utils::RandomExpandMatrix(*wox_, wox_->rows(), in, -init, init);
    utils::RandomExpandMatrix(*wcx_, wcx_->rows(), in, -init, init);
}

void LSTMParams::ResizeOutput(size_t out, double init) {
    size_t input_size = wix_->cols();
    utils::RandomExpandMatrix(*wix_, out, input_size, -init, init);
    utils::RandomExpandMatrix(*wih_, out, out, -init, init);
    utils::RandomExpandMatrix(*bi_, out, 1, -init, init);

    utils::RandomExpandMatrix(*wfx_, out, input_size, -init, init);
    utils::RandomExpandMatrix(*wfh_, out, out, -init, init);
    utils::RandomExpandMatrix(*bf_, out, 1, -init, init);

    utils::RandomExpandMatrix(*wox_, out, input_size, -init, init);
    utils::RandomExpandMatrix(*woh_, out, out, -init, init);
    utils::RandomExpandMatrix(*bo_, out, 1, -init, init);

    utils::RandomExpandMatrix(*wcx_, out, input_size, -init, init);
    utils::RandomExpandMatrix(*wch_, out, out, -init, init);
    utils::RandomExpandMatrix(*bc_, out, 1, -init, init);

    utils::RandomExpandMatrix(*cell_, out, 1, -init, init);
    utils::RandomExpandMatrix(*hidden_, out, 1, -init, init);
}

LSTMLayer::LSTMLayer(
        ComputationGraph& g, const LSTMParams& params, bool learnable)
    : wix_(g.CreateParam(params.wix_, learnable)),
    wih_(g.CreateParam(params.wih_, learnable)),
    bi_(g.CreateParam(params.bi_, learnable)),

    wfx_(g.CreateParam(params.wfx_, learnable)),
    wfh_(g.CreateParam(params.wfh_, learnable)),
    bf_(g.CreateParam(params.bf_, learnable)),

    wox_(g.CreateParam(params.wox_, learnable)),
    woh_(g.CreateParam(params.woh_, learnable)),
    bo_(g.CreateParam(params.bo_, learnable)),

    wcx_(g.CreateParam(params.wcx_, learnable)),
    wch_(g.CreateParam(params.wch_, learnable)),
    bc_(g.CreateParam(params.bc_, learnable)),
    hidden_(g.CreateParam(params.hidden_, learnable)),
    cell_(g.CreateParam(params.cell_, learnable)) {
}

Var LSTMLayer::Step(Var x) {
    Var input_gate = Sigmoid(wix_ * x + wih_ * hidden_ + bi_);
    Var forget_gate = Sigmoid(wfx_ * x + wfh_ * hidden_ + bf_);
    Var output_gate = Sigmoid(wox_ * x + woh_ * hidden_ + bo_);
    Var cell_write = Tanh(wcx_ * x + wch_ * hidden_ + bc_);

    Var cell_d = (forget_gate ^ cell_) + (input_gate ^ cell_write);

    Var hidden_d = output_gate ^ Tanh(cell_d);

    hidden_ = hidden_d;
    cell_ = cell_d;

    return hidden_;
}

} // nn
} // ad
