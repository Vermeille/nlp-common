#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"
#include "../helpers.h"

namespace ad {
namespace nn {

struct RNNLayerParams {
    std::shared_ptr<Param> whx_;
    std::shared_ptr<Param> whh_;
    std::shared_ptr<Param> bh_;
    std::shared_ptr<Param> h_;

    RNNLayerParams(int out_sz, int in_sz, double init = 1);

    void ResizeInput(int size, double init = 1) {
        utils::RandomExpandMatrix(whx_->value(), whx_->rows(), size, -init, init);
    }

    void ResizeOutput(int size, double init = 1) {
        utils::RandomExpandMatrix(whx_->value(), size, whx_->cols(), -init, init);
        utils::RandomExpandMatrix(whh_->value(), size, size, -init, init);
        utils::RandomExpandMatrix(bh_->value(), size, init, -init, init);
    }

    void Serialize(std::ostream& out) const;
    static RNNLayerParams FromSerialized(std::istream& in);
};

class RNNLayer {
    private:
        Var whx_;
        Var whh_;
        Var bh_;
        Var h_;

    public:
        RNNLayer(
                ComputationGraph& g,
                const RNNLayerParams& params,
                bool learnable = true);

        Var Step(Var in);
};


} // nn
} // ad
