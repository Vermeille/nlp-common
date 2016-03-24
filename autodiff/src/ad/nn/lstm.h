#pragma once

#include <memory>
#include <vector>

#include "../graph.h"

namespace ad {
namespace nn {

struct LSTMParams {
    std::shared_ptr<Param> wix_;
    std::shared_ptr<Param> wih_;
    std::shared_ptr<Param> bi_;

    std::shared_ptr<Param> wfx_;
    std::shared_ptr<Param> wfh_;
    std::shared_ptr<Param> bf_;

    std::shared_ptr<Param> wox_;
    std::shared_ptr<Param> woh_;
    std::shared_ptr<Param> bo_;
    // cell write params
    std::shared_ptr<Param> wcx_;
    std::shared_ptr<Param> wch_;
    std::shared_ptr<Param> bc_;

    std::shared_ptr<Param> cell_;
    std::shared_ptr<Param> hidden_;

    void ResizeInput(size_t in, double init = 1);
    void ResizeOutput(size_t out, double init = 1);

    LSTMParams(size_t output_size, size_t input_size);
};

class LSTMLayer {
    private:
        Var wix_;
        Var wih_;
        Var bi_;

        Var wfx_;
        Var wfh_;
        Var bf_;

        Var wox_;
        Var woh_;
        Var bo_;

        Var wcx_;
        Var wch_;
        Var bc_;

        Var hidden_;
        Var cell_;

    public:
        LSTMLayer(
                ComputationGraph& g,
                const LSTMParams& params,
                bool learnable = true);

        Var Step(Var in);
        void SetHidden(Var h) { hidden_ = h; }
        Var GetHidden() const { return hidden_; }

        std::vector<Var> Params() const {
            return {
                wix_, wih_, bi_,
                wfx_, wfh_, bf_,
                wox_, woh_, bo_,
                wcx_, wch_, bc_,
                hidden_,
                cell_
            };
        }
};

} // nn
} // ad
