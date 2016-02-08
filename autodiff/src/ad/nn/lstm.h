#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "neural-output.h"
#include "../graph.h"

namespace ad {
namespace nn {

class LSTMLayer {
    private:
        std::shared_ptr<Eigen::MatrixXd> wix_;
        std::shared_ptr<Eigen::MatrixXd> wih_;
        std::shared_ptr<Eigen::MatrixXd> bi_;

        std::shared_ptr<Eigen::MatrixXd> wfx_;
        std::shared_ptr<Eigen::MatrixXd> wfh_;
        std::shared_ptr<Eigen::MatrixXd> bf_;

        std::shared_ptr<Eigen::MatrixXd> wox_;
        std::shared_ptr<Eigen::MatrixXd> woh_;
        std::shared_ptr<Eigen::MatrixXd> bo_;
        // cell write params
        std::shared_ptr<Eigen::MatrixXd> wcx_;
        std::shared_ptr<Eigen::MatrixXd> wch_;
        std::shared_ptr<Eigen::MatrixXd> bc_;

    public:

    LSTMLayer(size_t output_size, size_t input_size);

    NeuralOutput<std::pair<std::vector<Var>, Var>> ComputeWithHidden(
            const NeuralOutput<std::vector<Var>>& in) const;

    NeuralOutput<std::vector<Var>> Compute(
            const NeuralOutput<std::vector<Var>>& in) const {
        auto res = ComputeWithHidden(in);
        return res.Forward(res.out.first, {});
    }

    NeuralOutput<Var> Encode(
            const NeuralOutput<std::vector<Var>>& in) const {
        auto res = ComputeWithHidden(in);
        return res.Forward(res.out.second, {});
    }

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

} // nn
} // ad
