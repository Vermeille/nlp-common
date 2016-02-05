#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"
#include "../helpers.h"
#include "neural-output.h"

namespace ad {
namespace nn {

class RNNLayer2 {
    private:
        std::shared_ptr<Eigen::MatrixXd> whx_;
        std::shared_ptr<Eigen::MatrixXd> whh_;
        std::shared_ptr<Eigen::MatrixXd> wox_;
        std::shared_ptr<Eigen::MatrixXd> woh_;
        std::shared_ptr<Eigen::MatrixXd> bo_;
        std::shared_ptr<Eigen::MatrixXd> bh_;

    public:
        RNNLayer2(int out_sz, int in_sz, int hidden_sz);

        NeuralOutput<std::pair<std::vector<Var>, Var>> Compute(
                NeuralOutput<std::vector<Var>> in) const;

        NeuralOutput<std::pair<std::vector<Var>, Var>> Compute(
                NeuralOutput<std::pair<std::vector<Var>, Var>> in) const {
            return Compute(in.Forward(in.out.first, {}));
        }
        void ResizeOutput(int size) {
            utils::RandomExpandMatrix(*wox_, size, wox_->cols(), -1, 1);
            utils::RandomExpandMatrix(*woh_, size, woh_->cols(), -1, 1);
            utils::RandomExpandMatrix(*bo_, size, 1, -1, 1);
        }

        void ResizeInput(int size) {
            utils::RandomExpandMatrix(*whx_, whx_->rows(), size, -1, 1);
            utils::RandomExpandMatrix(*wox_, wox_->rows(), size, -1, 1);
        }

        void ResizeHidden(int size) {
            utils::RandomExpandMatrix(*whx_, size, whx_->cols(), -1, 1);
            utils::RandomExpandMatrix(*whh_, size, size, -1, 1);
            utils::RandomExpandMatrix(*bh_, size, 1, -1, 1);
        }
};

} // nn
} // ad
