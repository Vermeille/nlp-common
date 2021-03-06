#pragma once

#include <memory>

#include "../graph.h"
#include "../helpers.h"

namespace ad {
namespace nn {

struct RNNParams {
    std::shared_ptr<Param> whx_;
    std::shared_ptr<Param> whh_;
    std::shared_ptr<Param> bh_;
    std::shared_ptr<Param> h_;

    RNNParams(int out_sz, int in_sz);

    void Serialize(std::ostream& out) const;
    static RNNParams FromSerialized(std::istream& in);
};

class RNNLayer {
    private:
        Var whx_;
        Var whh_;
        Var bh_;
        Var h_;

    public:
        RNNLayer(
                ComputationGraph& g,
                const RNNParams& params,
                bool learnable = true);

        Var Step(Var in);
        void SetHidden(Var h) { h_ = h; }
        Var GetHidden() const { return h_; }
};


} // nn
} // ad
