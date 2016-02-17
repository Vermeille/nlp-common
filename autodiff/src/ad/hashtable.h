#pragma once

#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include "graph.h"

namespace ad {
namespace nn {

class Hashtable {
    private:
        std::vector<std::shared_ptr<Param>> w_;
        size_t wordvec_size_;
        size_t vocab_size_;

    public:
        Hashtable(size_t wordvec_size, size_t vocab_size, double init = 1);

        std::string Serialize() const;
        static Hashtable FromSerialized(std::istream& in);

        void ResizeVectors(size_t size, double init = 1);
        void ResizeVocab(size_t size, double init = 1);

        Var MakeVarFor(
                ComputationGraph& g, size_t idx, bool learnable = true) const;
        std::shared_ptr<Param> Get(size_t idx) const;
};

} // nn
} // ad
