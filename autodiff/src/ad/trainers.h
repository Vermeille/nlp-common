#pragma once

#include <Eigen/Dense>

namespace ad {
namespace train {

template <class Updater>
class FeedForwardTrainer {
    private:
        Updater updater_;
    public:
        FeedForwardTrainer(const Updater& updater)
            : updater_(updater) {
        }

        template <class Model, class T, class U>
        double Step(ComputationGraph& g, Model& model,
                const T& input, const U& out) {
            Var h = model.Step(g, input);
            Var J = model.Cost(g, h, out);

            g.BackpropFrom(J);
            g.Update(updater_);

            return J.value()(0, 0);
        }
};

} // train
} // ad
