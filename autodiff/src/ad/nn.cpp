#include "nn.h"

#include "operators.h"

namespace ad {
namespace nn {

Var L2(const Var& x) {
    return Mean(EltSquare(x));
}

Var L2(std::vector<Var> x) {
    Var sum = L2(x[0]);
    for (size_t i = 1; i < x.size(); ++i) {
        sum = sum + L2(x[1]);
    }
    return sum;
}

} // nn
} // ad
