#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"
#include "../helpers.h"

namespace ad {
namespace nn {

struct RNNLayerParams {
    std::shared_ptr<Eigen::MatrixXd> whx_;
    std::shared_ptr<Eigen::MatrixXd> whh_;
    std::shared_ptr<Eigen::MatrixXd> bh_;
    std::shared_ptr<Eigen::MatrixXd> h_;

    RNNLayerParams(int out_sz, int in_sz, double init = 1);

    void ResizeInput(int size, double init = 1) {
        utils::RandomExpandMatrix(*whx_, whx_->rows(), size, -init, init);
    }

    void ResizeOutput(int size, double init = 1) {
        utils::RandomExpandMatrix(*whx_, size, whx_->cols(), -init, init);
        utils::RandomExpandMatrix(*whh_, size, size, -init, init);
        utils::RandomExpandMatrix(*bh_, size, init, -init, init);
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
        RNNLayer(ComputationGraph& g, const RNNLayerParams& params);

        Var Step(Var in);
};


} // nn
} // ad
