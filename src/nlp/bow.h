#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>
#include <ad/ad.h>

#include "document.h"

class BagOfWords {
    ad::nn::Hashtable words_;
    size_t output_size_;

    ad::nn::NeuralOutput<ad::Var> ComputeModel(
            ad::ComputationGraph& g,
            const std::vector<WordFeatures>& ws) const;

  public:
    BagOfWords(size_t in_sz, size_t out_sz);
    BagOfWords();

    double weights(size_t label, size_t word) const;

    std::string Serialize() const;
    static BagOfWords FromSerialized(std::istream& file);

    Eigen::MatrixXd ComputeClass(const std::vector<WordFeatures>& ws) const;

    int Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

