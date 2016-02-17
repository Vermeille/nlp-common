#pragma once

#include <memory>

#include "../graph.h"

namespace ad {
namespace nn {

struct FullyConnParams {
    std::shared_ptr<Param> w_;
    std::shared_ptr<Param> b_;

    FullyConnParams(int out_sz, int in_sz, double init = 1);

    void ResizeOutput(int size, double init = 1);
    void ResizeInput(int size, double init = 1);

    void Serialize(std::ostream& out) const;
    static FullyConnParams FromSerialized(std::istream& in);
};

class FullyConnLayer {
    private:
        Var w_;
        Var b_;

    public:
        FullyConnLayer(
                ComputationGraph& g,
                const FullyConnParams& params,
                bool learnable = true);

        Var Compute(Var in) const;

        std::vector<Var> Params() const;
};

} // nn
} // ad

