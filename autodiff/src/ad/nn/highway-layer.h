#pragma once

#pragma once

#include <memory>

#include <Eigen/Dense>

#include "../graph.h"

namespace ad {
namespace nn {

struct HighwayLayerParams {
    std::shared_ptr<Param> w_;
    std::shared_ptr<Param> wt_;
    std::shared_ptr<Param> wc_;

    HighwayLayerParams(size_t sz, double init = 1);
    void Resize(size_t in, double init = 1);
};

class HighwayLayer {
    private:
        Var w_;
        Var wt_;
        Var wc_;

    public:
        HighwayLayer(
                ComputationGraph& g,
                const HighwayLayerParams& params,
                bool learnable = true);
        Var Step(Var x) const;
};

} // nn
} // ad
