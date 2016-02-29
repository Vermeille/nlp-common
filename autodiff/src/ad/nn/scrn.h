#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../graph.h"
#include "../operators.h"

namespace ad {
namespace nn {

struct SCRNParams {
    std::shared_ptr<Param> alpha_;
    std::shared_ptr<Param> wsx_;

    std::shared_ptr<Param> whs_;
    std::shared_ptr<Param> whx_;
    std::shared_ptr<Param> whh_;

    std::shared_ptr<Param> ctx_;
    std::shared_ptr<Param> hidden_;

    SCRNParams(size_t ctx_size, size_t hidden_size, size_t input_size)
        : alpha_(std::make_shared<Param>(ctx_size, 1, Constant(3))),
        wsx_(std::make_shared<Param>(ctx_size, input_size)),
        whs_(std::make_shared<Param>(hidden_size, ctx_size)),
        whx_(std::make_shared<Param>(hidden_size, input_size)),
        whh_(std::make_shared<Param>(hidden_size, hidden_size)),
        ctx_(std::make_shared<Param>(ctx_size, 1, Gaussian(0, 0.1))),
        hidden_(std::make_shared<Param>(hidden_size, 1, Gaussian(0, 0.1))) {
    }
};

class SCRNLayer {
    private:
        Var alpha_;
        Var wsx_;

        Var whs_;
        Var whx_;
        Var whh_;

        Var ctx_;
        Var hidden_;

    public:
        SCRNLayer(ComputationGraph& g,
                const SCRNParams& params,
                bool learnable = true)
            : alpha_(g.CreateParam(params.alpha_, learnable)),
            wsx_(g.CreateParam(params.wsx_, learnable)),
            whs_(g.CreateParam(params.whs_, learnable)),
            whx_(g.CreateParam(params.whx_, learnable)),
            whh_(g.CreateParam(params.whh_, learnable)),
            ctx_(g.CreateParam(params.ctx_, learnable)),
            hidden_(g.CreateParam(params.hidden_, learnable)) {
        }

        Var Step(Var x) {
            Var gate = Sigmoid(alpha_); // contrain weigths in [0; 1]
            ctx_ = ((1 - gate) ^ (wsx_ * x)) + (gate ^ ctx_);
            hidden_ = Tanh(whs_ * ctx_ + whx_ * x + whh_ * hidden_);
            return ColAppend(ctx_, hidden_);
        }

        std::vector<Var> Params() const {
            return {
                alpha_, wsx_,
                whs_, whx_, whh_,
                ctx_, hidden_
            };
        }
};

} // nn
} // ad

