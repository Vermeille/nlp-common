#pragma once

#include <utility>
#include <memory>
#include <set>
#include <vector>

#include "graph.h"
#include "helpers.h"
#include "operators.h"
#include "hashtable.h"

#include "nn/neural-output.h"
#include "nn/input-layer.h"
#include "nn/fullyconn.h"
#include "nn/rnn.h"
#include "nn/lstm.h"

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

NeuralOutput<std::vector<Var>> HashtableQuery(
        ComputationGraph& g,
        const Hashtable& vecs,
        const std::vector<int>& idx);

NeuralOutput<Var> HashtableQuery(
        ComputationGraph& g, const Hashtable& vecs, int& idx);

template <class F>
NeuralOutput<std::vector<Var>> Map(F&& f, const NeuralOutput<std::vector<Var>>& in) {
    std::vector<Var> out;
    std::transform(in.out.begin(), in.out.end(), std::back_inserter(out), f);
    return in.Forward(out, {});
}

template <class F>
NeuralOutput<Var> Map(F&& f, const NeuralOutput<Var>& in) {
    return in.Forward(f(in.out), {});
}

template <class F>
NeuralOutput<Var> Fold(const NeuralOutput<std::vector<Var>>& in, F&& f) {
    if (in.out.empty()) {
        throw std::runtime_error("Can't fold an empty sequence");
    }

    return in.Forward(
            std::accumulate(in.out.begin() + 1, in.out.end(), in.out[0], f),
            {});
}

NeuralOutput<Var> Sum(const NeuralOutput<std::vector<Var>>& in);

template <class BinaryOp>
NeuralOutput<std::vector<Var>> ZipWith(
        NeuralOutput<std::vector<Var>>& xs,
        NeuralOutput<std::vector<Var>>& ys,
        BinaryOp op) {
    if (xs.out.size() != ys.out.size()) {
        throw std::runtime_error("can't zip sequences of different lengths");
    }

    std::vector<Var> out;
    out.reserve(xs.out.size());
    for (size_t i = 0; i < xs.out.size(); ++i) {
        out.push_back(op(xs.out[i], ys.out[i]));
    }
    return xs.Forward(out, *ys.params);
}

NeuralOutput<std::vector<Var>> Sum(
        NeuralOutput<std::vector<Var>>& xs,
        NeuralOutput<std::vector<Var>>& ys);

} // nn
} // ad
