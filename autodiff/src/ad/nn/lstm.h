#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../graph.h"

namespace ad {
namespace nn {

struct LSTMParams {
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

    std::shared_ptr<Eigen::MatrixXd> cell_;
    std::shared_ptr<Eigen::MatrixXd> hidden_;

    void ResizeInput(size_t in, double init = 1);
    void ResizeOutput(size_t out, double init = 1);

    LSTMParams(size_t output_size, size_t input_size, double init = 1);
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
};

} // nn
} // ad
