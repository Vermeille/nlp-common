#include "nn.h"

#include "operators.h"

namespace ad {
namespace nn {

Var L2(const Var& x) {
    return Mean(EltSquare(x));
}

} // nn
} // ad
