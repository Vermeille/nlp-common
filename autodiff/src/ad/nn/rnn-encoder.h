#pragma once

#include "neural-output.h"
#include "../helpers.h"

namespace ad {
namespace nn {

class RNNEncoderLayer {
    private:
        std::shared_ptr<Eigen::MatrixXd> whx_;
        std::shared_ptr<Eigen::MatrixXd> whh_;
        std::shared_ptr<Eigen::MatrixXd> bh_;

    public:
        RNNEncoderLayer(int out_sz, int in_sz);

        NeuralOutput<Var> Compute(NeuralOutput<std::vector<Var>> in) const;

        NeuralOutput<Var> Compute(
                NeuralOutput<std::pair<std::vector<Var>, Var>> in) const {
            return Compute(in.Forward(in.out.first, {}));
        }

        void ResizeOutput(int size) {
            utils::RandomExpandMatrix(*whx_, size, whx_->cols(), -1, 1);
            utils::RandomExpandMatrix(*whh_, size, size, -1, 1);
            utils::RandomExpandMatrix(*bh_, size, 1, -1, 1);
        }

        void ResizeInput(int size) {
            utils::RandomExpandMatrix(*whx_, whx_->rows(), size, -1, 1);
        }
};

} // nn
} // ad
