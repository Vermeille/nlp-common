#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"
#include "neural-output.h"

namespace ad {
namespace nn {

class FullyConnLayer {
    private:
        std::shared_ptr<Eigen::MatrixXd> w_;
        std::shared_ptr<Eigen::MatrixXd> b_;

    public:
        FullyConnLayer(int out_sz, int in_sz);
        FullyConnLayer(
                std::shared_ptr<Eigen::MatrixXd> w,
                std::shared_ptr<Eigen::MatrixXd> b);

        NeuralOutput<Var> Compute(NeuralOutput<Var> in);

        Eigen::MatrixXd& w() { return *w_; }
        const Eigen::MatrixXd& w() const { return *w_; }

        Eigen::MatrixXd& b() { return *b_; }
        const Eigen::MatrixXd& b() const { return *b_; }

        void ResizeOutput(int size);
        void ResizeInput(int size);

        void Serialize(std::ostream& out) const;
        static FullyConnLayer FromSerialized(std::istream& in);
};

} // nn
} // ad

