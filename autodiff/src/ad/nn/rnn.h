#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"
#include "../helpers.h"
#include "neural-output.h"

namespace ad {
namespace nn {

class RNNLayer {
    private:
        std::shared_ptr<Eigen::MatrixXd> whx_;
        std::shared_ptr<Eigen::MatrixXd> whh_;
        std::shared_ptr<Eigen::MatrixXd> bh_;

    public:
        RNNLayer(int out_sz, int in_sz);

        NeuralOutput<std::pair<std::vector<Var>, Var>> ComputeWithHidden(
                NeuralOutput<std::vector<Var>> in) const;

        NeuralOutput<std::vector<Var>> Compute(
                NeuralOutput<std::vector<Var>> in) const {
            auto res = ComputeWithHidden(in);
            return res.Forward(res.out.first, {});
        }

        NeuralOutput<Var>
            Encode(const NeuralOutput<std::vector<Var>>& in) const {
            auto res = ComputeWithHidden(in);
            return res.Forward(res.out.second, {});
        }

        void ResizeInput(int size) {
            utils::RandomExpandMatrix(*whx_, whx_->rows(), size, -1, 1);
        }

        void ResizeOutput(int size) {
            utils::RandomExpandMatrix(*whx_, size, whx_->cols(), -1, 1);
            utils::RandomExpandMatrix(*whh_, size, size, -1, 1);
            utils::RandomExpandMatrix(*bh_, size, 1, -1, 1);
        }

        void Serialize(std::ostream& out) const;
        static RNNLayer FromSerialized(std::istream& in);
};


} // nn
} // ad
