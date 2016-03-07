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
    ad::opt::Adagrad adagrad_;
    ad::train::FeedForwardTrainer trainer_;

  public:
    BagOfWords(size_t in_sz, size_t out_sz);
    BagOfWords();

    double weights(size_t label, size_t word) const;

    std::string Serialize() const;
    static BagOfWords FromSerialized(std::istream& file);

    ad::Var Step(
            ad::ComputationGraph& g,
            const std::vector<WordFeatures>& ws) const;

    ad::Var Cost(ad::ComputationGraph& g, ad::Var h, int output_class);

    Eigen::MatrixXd ComputeClass(const std::vector<WordFeatures>& ws) const;

    double Train(const Document& doc);
    double Test(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

