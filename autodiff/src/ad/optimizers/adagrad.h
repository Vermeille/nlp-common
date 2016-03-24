#pragma once

#include <map>
#include <utility>

#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Adagrad : public Optimizer {
    private:
        std::map<int, Matrix> grad_acc_;
        double learning_rate_;
        double epsilon_;

        Matrix& AccumulatedGradient(Var v);

    public:
        Adagrad(double learning_rate = 1, double epsilon = 1e-5) :
            learning_rate_(learning_rate),
            epsilon_(epsilon) {
        }

        virtual void Update(Var& v);
};

} // opt
} // ad
