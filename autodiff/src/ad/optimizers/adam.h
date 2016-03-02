#pragma once

#include <map>
#include <utility>

#include <Eigen/Dense>
#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Adam : public Optimizer {
    private:
        std::map<int, Eigen::MatrixXd> m_;
        std::map<int, Eigen::MatrixXd> v_;
        double learning_rate_;
        double beta1_;
        double beta2_;
        double epsilon_;
        size_t t_;

        Eigen::MatrixXd& GetM(Var v);
        Eigen::MatrixXd& GetV(Var v);

    public:
        Adam(
                double learning_rate = 0.001,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double epsilon = 1e-8) :
            learning_rate_(learning_rate),
            beta1_(beta1),
            beta2_(beta2),
            epsilon_(epsilon),
            t_(0) {
        }

        virtual void Update(Var& v);
};

} // opt
} // ad
