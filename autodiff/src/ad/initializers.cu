#include "initializers.h"
#include "helpers.h"

#include "kernels/kernels.h"

#include <cmath>
#include <curand.h>

namespace ad {

static int i = 0;

static curandGenerator_t* GetUniformGenerator() {
    static curandGenerator_t* gen = nullptr;
    if (!gen) {
        gen = new curandGenerator_t;
        curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(*gen, 1234ULL);

        std::atexit([&]() {
                curandDestroyGenerator(*gen);
            });
    }
    return gen;
}

void Gaussian::Init(Matrix& mat) const {
    curandGenerator_t* gen = GetUniformGenerator();
    curandGenerateNormal(*gen, mat.data().Get(), mat.size(), mu_, sigma_);
    std::cout << "gaussian inited " << i << "\n";
    ++i;
}

void Uniform::Init(Matrix& mat) const {
    curandGenerator_t* gen = GetUniformGenerator();
    curandGenerateUniform(*gen, mat.data().Get(), mat.size());
    auto x = cuda::Array(mat.data());
    auto scale = cuda::Value(std::fabs(from_) + std::fabs(to_));
    auto from = cuda::Value(from_);
    cuda::RunKernel(cuda::Seq(
            x = x * scale + from),
        mat.size());
    std::cout << "uniform inited " << i << "\n";
    ++i;
}

} // ad
