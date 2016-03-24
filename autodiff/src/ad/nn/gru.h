#pragma once

#include <memory>
#include <vector>

#include "../graph.h"

namespace ad {
namespace nn {

struct GRUParams {
    std::shared_ptr<Param> wzx_;
    std::shared_ptr<Param> wzh_;
    std::shared_ptr<Param> bz_;

    std::shared_ptr<Param> wrx_;
    std::shared_ptr<Param> wrh_;
    std::shared_ptr<Param> br_;

    std::shared_ptr<Param> whx_;
    std::shared_ptr<Param> whh_;
    std::shared_ptr<Param> bh_;

    std::shared_ptr<Param> hidden_;

    void ResizeInput(size_t in, double init = 1);
    void ResizeOutput(size_t out, double init = 1);

    GRUParams(size_t output_size, size_t input_size);
};

class GRULayer {
    private:
        Var wzx_;
        Var wzh_;
        Var bz_;

        Var wrx_;
        Var wrh_;
        Var br_;

        Var whx_;
        Var whh_;
        Var bh_;

        Var hidden_;

    public:
        GRULayer(
                ComputationGraph& g,
                const GRUParams& params,
                bool learnable = true);

        Var Step(Var in);
        void SetHidden(Var h) { hidden_ = h; }
        Var GetHidden() const { return hidden_; }

        std::vector<Var> Params() const {
            return {
                wzx_, wzh_, bz_,
                wrx_, wrh_, br_,
                whx_, whh_, bh_,
                hidden_,
            };
        }
};

} // nn
} // ad
