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

    nn::FullyConnLayer fc(1, 2);

    for (int i = 0; i < 100; ++ i) {
        ComputationGraph g;
        auto dataset = GenDataset(nb_examples);

        Var x = g.CreateParam(dataset.first);
        Var y = g.CreateParam(dataset.second);

        auto input = nn::InputLayer(x);
        auto output = fc.Compute(input);
        Var j = MSE(output.out, y) + 0.001 * nn::L2ForAllParams(output);

        std::cout << "COST = " << j.value() << "\n";

        opt::SGD sgd(0.1 / nb_examples);
        g.BackpropFrom(j);
        g.Update(sgd, *output.params);
    }

    std::cout << "w = " << fc.w() << "\nb = " << fc.b() << "\n";

    return 0;
}

