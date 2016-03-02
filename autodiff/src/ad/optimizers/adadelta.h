#pragma once

#include <map>
#include <utility>

#include <Eigen/Dense>
#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Adadelta : public Optimizer {
    private:
        std::map<int, Eigen::MatrixXd> r_;
        std::map<int, Eigen::MatrixXd> s_;
        double learning_rate_;
        double rho_;
        double epsilon_;

        Eigen::MatrixXd& GetR(Var v);
        Eigen::MatrixXd& GetS(Var v);

    public:
        Adadelta(double learning_rate = 1, double rho = 0.95, double epsilon = 1e-6) :
            learning_rate_(learning_rate),
            rho_(rho),
            epsilon_(epsilon) {
        }

        virtual void Update(Var& v);
};

} // opt
} // ad
