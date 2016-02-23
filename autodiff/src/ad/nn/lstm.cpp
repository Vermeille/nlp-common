#include "lstm.h"

#include "../operators.h"
#include "../helpers.h"

namespace ad {
namespace nn {

LSTMParams::LSTMParams(size_t output_size, size_t input_size)
    : wix_(std::make_shared<Param>(output_size, input_size, Xavier())),
    wih_(std::make_shared<Param>(output_size, output_size, Xavier())),
    bi_(std::make_shared<Param>(output_size, 1, Constant(0))),

    wfx_(std::make_shared<Param>(output_size, input_size, Xavier())),
    wfh_(std::make_shared<Param>(output_size, output_size, Xavier())),
    bf_(std::make_shared<Param>(output_size, 1, Constant(5))),

    wox_(std::make_shared<Param>(output_size, input_size, Xavier())),
    woh_(std::make_shared<Param>(output_size, output_size, Xavier())),
    bo_(std::make_shared<Param>(output_size, 1, Constant(0))),

    wcx_(std::make_shared<Param>(output_size, input_size, Xavier())),
    wch_(std::make_shared<Param>(output_size, output_size, Xavier())),
    bc_(std::make_shared<Param>(output_size, 1, Constant(0))),

    cell_(std::make_shared<Param>(output_size, 1, Gaussian(0, 0.1))),
    hidden_(std::make_shared<Param>(output_size, 1, Gaussian(0, 0.1))) {
}


void LSTMParams::ResizeInput(size_t in, double init) {
    utils::RandomExpandMatrix(wix_->value(), wix_->rows(), in, -init, init);
    utils::RandomExpandMatrix(wfx_->value(), wfx_->rows(), in, -init, init);
    utils::RandomExpandMatrix(wox_->value(), wox_->rows(), in, -init, init);
    utils::RandomExpandMatrix(wcx_->value(), wcx_->rows(), in, -init, init);
}

void LSTMParams::ResizeOutput(size_t out, double init) {
    size_t input_size = wix_->cols();
    utils::RandomExpandMatrix(wix_->value(), out, input_size, -init, init);
    utils::RandomExpandMatrix(wih_->value(), out, out, -init, init);
    utils::RandomExpandMatrix(bi_->value(), out, 1, -init, init);

    utils::RandomExpandMatrix(wfx_->value(), out, input_size, -init, init);
    utils::RandomExpandMatrix(wfh_->value(), out, out, -init, init);
    utils::RandomExpandMatrix(bf_->value(), out, 1, -init, init);

    utils::RandomExpandMatrix(wox_->value(), out, input_size, -init, init);
    utils::RandomExpandMatrix(woh_->value(), out, out, -init, init);
    utils::RandomExpandMatrix(bo_->value(), out, 1, -init, init);

    utils::RandomExpandMatrix(wcx_->value(), out, input_size, -init, init);
    utils::RandomExpandMatrix(wch_->value(), out, out, -init, init);
    utils::RandomExpandMatrix(bc_->value(), out, 1, -init, init);

    utils::RandomExpandMatrix(cell_->value(), out, 1, -init, init);
    utils::RandomExpandMatrix(hidden_->value(), out, 1, -init, init);
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
