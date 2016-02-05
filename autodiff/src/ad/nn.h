#pragma once

#include <utility>
#include <memory>
#include <set>
#include <vector>

#include "graph.h"
#include "helpers.h"
#include "operators.h"

#include "nn/neural-output.h"
#include "nn/input-layer.h"
#include "nn/fullyconn.h"
#include "nn/rnn1.h"
#include "nn/rnn2.h"
#include "nn/rnn-encoder.h"

namespace ad {
namespace nn {

Var L2(const Var& x);

template <class T>
Var L2ForAllParams(NeuralOutput<T> in) {
    if (in.params->empty()) {
        throw std::runtime_error("no parameters in the L2 layer");
    }

    auto i = in.params->begin();
    Var sum = L2(*i);
    ++i;
    for (/* empty */; i != in.params->end(); ++i) {
        sum = sum + L2(*i);
    }
    return sum;
}


} // nn
} // ad
