#include "initializers.h"
#include "helpers.h"

#include <random>

namespace ad {

void Gaussian::Init(Eigen::MatrixXd& mat) const {
    std::default_random_engine generator;
    std::normal_distribution<double> gaussian(mu_, sigma_);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            mat(i, j) = gaussian(generator);
        }
    }
}

void Uniform::Init(Eigen::MatrixXd& mat) const {
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            mat(i, j) = utils::RandomRange(from_, to_);
        }
    }
}

} // ad
