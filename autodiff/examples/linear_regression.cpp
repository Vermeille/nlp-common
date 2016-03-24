#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

#include <ad/ad.h>

std::pair<ad::RWMatrix, ad::RWMatrix> GenDataset(int nb_examples) {
    // dataset
    ad::RWMatrix x(2, nb_examples);
    ad::RWMatrix y(1, nb_examples);
    for (int k = 0; k < nb_examples; ++k) {
        float x1 = (rand() % 30 - 15) / 15.;
        float x2 = (rand() % 30 - 15) / 15.;
        x(0, k) = x1;
        x(1, k) = x2;
        y(0, k) =  x1 * 8 + x2 * 3 + 5;
    }
    return std::make_pair(std::move(x), std::move(y));
}

int main() {
    using namespace ad;
    const int nb_examples = 1;

    nn::FullyConnParams fc_params(1, 2);
    ad::opt::Adam adagrad;

    for (int i = 0; i < 10000; ++ i) {
        ComputationGraph g;
        nn::FullyConnLayer fc(g, fc_params);
        auto dataset = GenDataset(nb_examples);

        Var x = g.CreateParam(ad::Matrix(dataset.first));
        Var y = g.CreateParam(ad::Matrix(dataset.second));

        auto output = fc.Compute(x);
        Var j = MSE(output, y) + 0.001 * nn::L2(g.GetAllLearnableVars());

        g.BackpropFrom(j);
        g.Update(adagrad);
        std::cout << "COST = " << j.value().CudaRead(0, 0) << "\n";
    }

    std::cout << "w = " << fc_params.w_->value().Fetch() << "\n"
        "b = " << fc_params.b_->value().Fetch() << "\n";

    return 0;
}

