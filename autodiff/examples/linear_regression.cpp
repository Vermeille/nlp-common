#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

#include <ad/ad.h>

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GenDataset(int nb_examples) {
    // dataset
    Eigen::MatrixXd x(2, nb_examples);
    Eigen::MatrixXd y(1, nb_examples);
    for (int k = 0; k < nb_examples; ++k) {
        float x1 = (rand() % 30 - 15) / 15.;
        float x2 = (rand() % 30 - 15) / 15.;
        x(0, k) = x1;
        x(1, k) = x2;
        y(0, k) =  x1 * 8 + x2 * 3 + 5;
    }
    return std::make_pair(x, y);
}

int main() {
    using namespace ad;
    const int nb_examples = 1;

    nn::FullyConnParams fc_params(1, 2);
    ad::opt::Adagrad adagrad;

    for (int i = 0; i < 100; ++ i) {
        ComputationGraph g;
        nn::FullyConnLayer fc(g, fc_params);
        auto dataset = GenDataset(nb_examples);

        Var x = g.CreateParam(dataset.first);
        Var y = g.CreateParam(dataset.second);

        auto output = fc.Compute(x);
        Var j = MSE(output, y) + 0.001 * nn::L2(g.GetAllLearnableVars());

        std::cout << "COST = " << j.value() << "\n";

        opt::SGD sgd(0.1 / nb_examples);
        g.BackpropFrom(j);
        g.Update(adagrad);
    }

    std::cout << "w = " << fc_params.w_->value() << "\n"
        "b = " << fc_params.b_->value() << "\n";

    return 0;
}

