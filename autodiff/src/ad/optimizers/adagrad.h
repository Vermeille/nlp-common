#pragma once

#include <map>
#include <utility>

#include <Eigen/Dense>
#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Adagrad : public Optimizer {
    private:
        std::map<int, Eigen::MatrixXd> grad_acc_;
        double learning_rate_;
        double epsilon_;

        Eigen::MatrixXd& AccumulatedGradient(Var v);

    public:
        Adagrad(double learning_rate = 1, double epsilon = 1e-5) :
            learning_rate_(learning_rate),
            epsilon_(epsilon) {
        }

        virtual void Update(Var& v);
};

} // opt
} // ad
