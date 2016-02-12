#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"

namespace ad {
namespace nn {

struct FullyConnParams {
    std::shared_ptr<Eigen::MatrixXd> w_;
    std::shared_ptr<Eigen::MatrixXd> b_;

    FullyConnParams(int out_sz, int in_sz, double init = 1);

    void ResizeOutput(int size, double init = 1);
    void ResizeInput(int size, double init = 1);

    void Serialize(std::ostream& out) const;
    static FullyConnParams FromSerialized(std::istream& in);

    Eigen::MatrixXd& w() { return *w_; }
    const Eigen::MatrixXd& w() const { return *w_; }

    Eigen::MatrixXd& b() { return *b_; }
    const Eigen::MatrixXd& b() const { return *b_; }
};

class FullyConnLayer {
    private:
        Var w_;
        Var b_;

    public:
        FullyConnLayer(
                ComputationGraph& g,
                const FullyConnParams& params,
                bool learnable = true);

        Var Compute(Var in) const;
};

} // nn
} // ad

