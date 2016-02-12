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
        std::vector<std::shared_ptr<Eigen::MatrixXd>> w_;
        size_t wordvec_size_;
        size_t vocab_size_;

    public:
        Hashtable(size_t wordvec_size, size_t vocab_size);

        std::string Serialize() const;
        static Hashtable FromSerialized(std::istream& in);

        void ResizeVectors(size_t size);
        void ResizeVocab(size_t size);

        Var MakeVarFor(
                ComputationGraph& g, size_t idx, bool learnable = true) const;
        std::shared_ptr<Eigen::MatrixXd> Get(size_t idx) const;
};

} // nn
} // ad
