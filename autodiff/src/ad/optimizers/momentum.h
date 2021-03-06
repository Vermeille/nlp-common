#pragma once

#include <map>
#include <utility>

#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Momentum : public Optimizer {
    private:
        std::map<int, Matrix> velocity_;
        double learning_rate_;
        double momentum_;

        Matrix& AccumulatedVelocity(Var v);

    public:
        Momentum(double learning_rate = 0.1, double momentum = 0.9) :
            learning_rate_(learning_rate),
            momentum_(momentum) {
        }

        virtual void Update(Var& v);
};

} // opt
} // ad
