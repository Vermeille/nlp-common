#pragma once

#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"
#include "neural-output.h"

namespace ad {
namespace nn {

class HighwayLayer {
    private:
        std::shared_ptr<Eigen::MatrixXd> w_;
        std::shared_ptr<Eigen::MatrixXd> wt_;
        std::shared_ptr<Eigen::MatrixXd> wc_;

    public:
        HighwayLayer(size_t out, size_t in);

        NeuralOutput<Var> Compute(NeuralOutput<Var> in) const;

        void ResizeInput(size_t in);
        void ResizeOutput(size_t in);
};

} // nn
} // ad
