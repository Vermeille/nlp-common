#pragma once

#include <map>
#include <utility>

#include "../optimizer.h"
#include "../graph.h"

namespace ad {
namespace opt {

class Adam : public Optimizer {
    private:
        std::map<int, Matrix> m_;
        std::map<int, Matrix> v_;
        float learning_rate_;
        float beta1_;
        float beta2_;
        float epsilon_;
        size_t t_;

        Matrix& GetM(Var v);
        Matrix& GetV(Var v);

    public:
        Adam(
                float learning_rate = 0.001,
                float beta1 = 0.9,
                float beta2 = 0.999,
                float epsilon = 1e-6) :
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
