#pragma once

#include <memory>
#include <set>

#include "../graph.h"

namespace ad {
namespace nn {

template <class T>
struct NeuralOutput {
    T out;
    std::shared_ptr<std::set<Var>> params;

    template <class U>
    NeuralOutput<U> Forward(const U& new_out, const std::vector<Var>& new_params) {
        for (Var p : new_params) {
            params->insert(p);
        }
        return {new_out, params};
    }
};

} // nn
} // ad
