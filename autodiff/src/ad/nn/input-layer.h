#pragma once

#include "neural-output.h"

namespace ad {
namespace nn {

template <class T>
NeuralOutput<T> InputLayer(const T& x) {
    return NeuralOutput<T>{x, std::make_shared<std::set<Var>>()};
}


} // nn
} // ad
