#pragma once

#include <map>
#include <utility>

#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Adadelta : public Optimizer {
    private:
        std::map<int, Matrix> r_;
        std::map<int, Matrix> s_;
        float learning_rate_;
        float rho_;
        float epsilon_;

        Matrix& GetR(Var v);
        Matrix& GetS(Var v);

    public:
        Adadelta(float learning_rate = 1, float rho = 0.95, float epsilon = 1e-4f) :
            learning_rate_(learning_rate),
            rho_(rho),
            epsilon_(epsilon) {
        }

        virtual void Update(Var& v);
};

} // opt
} // ad
