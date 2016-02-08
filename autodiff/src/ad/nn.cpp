#include "nn.h"

#include "operators.h"

namespace ad {
namespace nn {

Var L2(const Var& x) {
    return Mean(EltSquare(x));
}

NeuralOutput<std::vector<Var>> HashtableQuery(
        ComputationGraph& g,
        const Hashtable& vecs,
        const std::vector<int>& idxs) {
    std::vector<Var> out;
    out.reserve(idxs.size());

    for (auto& i : idxs) {
        out.push_back(vecs.MakeVarFor(g, i));
    }

    auto layer = InputLayer(out);

    for (auto& w : out) {
        layer.params->insert(w);
    }
    return layer;
}

NeuralOutput<Var> HashtableQuery(
        ComputationGraph& g,
        const Hashtable& vecs,
        int& idx) {
    Var vec = vecs.MakeVarFor(g, idx);
    auto layer = InputLayer(vec);
    layer.params->insert(vec);
    return layer;
}

NeuralOutput<Var> Sum(const NeuralOutput<std::vector<Var>>& in) {
    return Fold(in, [](const Var& a, const Var& b) { return a + b; });
}

NeuralOutput<std::vector<Var>> Sum(
        NeuralOutput<std::vector<Var>>& xs,
        NeuralOutput<std::vector<Var>>& ys) {
    return ZipWith(xs, ys, [](const Var& a, const Var& b){ return a + b; });
}

std::vector<NeuralOutput<Var>> Lift(const NeuralOutput<std::vector<Var>>& in) {
    std::vector<NeuralOutput<Var>> out;
    out.reserve(in.out.size());
    for (Var v : in.out) {
        out.push_back(in.Forward(v, {}));
    }
    return out;
}

} // nn
} // ad
