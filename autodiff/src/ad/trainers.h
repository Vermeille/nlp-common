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

        template <class ModelGen, class T, class U>
        double Step(const T& input, const U& out, const ModelGen& model_gen) {
            ComputationGraph g;
            auto model = model_gen(g);
            Var h = model.Step(g, input);
            Var J = model.Cost(g, h, out);

            g.BackpropFrom(J);
            g.Update(updater_);

            return J.value()(0, 0);
        }
};

template <class Updater>
class WholeSequenceTaggerTrainer {
    private:
        Updater updater_;

    public:
        WholeSequenceTaggerTrainer(const Updater& updater)
            : updater_(updater) {
        }

        template <class ModelGen, class T, class U>
        double Step(const T& input, const U& expected, const ModelGen& model_gen) {
            if (input.empty()) {
                return 0;
            }

            std::vector<Var> predicted;
            ComputationGraph g;
            auto model = model_gen(g);
            for (auto in : input) {
                predicted.push_back(model.Step(g, in));
            }

            Var J = model.Cost(g, predicted, expected);

            g.BackpropFrom(J);
            g.Update(updater_);

            return J.value()(0, 0);
        }
};

template <class Updater>
class IterativeSequenceTaggerTrainer {
    private:
        Updater updater_;

    public:
        IterativeSequenceTaggerTrainer(const Updater& updater)
            : updater_(updater) {
        }

        template <class ModelGen, class T, class U>
        double Step(const T& input, const U& expected, const ModelGen& model_gen) {
            if (input.empty()) {
                return 0;
            }

            double nll = 0;
            Var h(nullptr);
            for (size_t last = 0; last < input.size(); ++last) {
                ComputationGraph g;
                auto model = model_gen(g);
                for (size_t i = 0; i <= last; ++i) {
                    h = model.Step(g, input[i]);
                }

                Var J = model.Cost(g, h, expected[last]);

                nll += J.value()(0, 0);
                g.BackpropFrom(J);
                g.Update(updater_);
            }

            return nll / input.size();
        }
};

} // train
} // ad
