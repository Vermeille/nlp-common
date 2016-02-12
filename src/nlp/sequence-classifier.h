#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>
#include <ad/ad.h>

#include "document.h"

class SequenceClassifier {
    ad::nn::Hashtable words_;
    ad::nn::RNNLayerParams encoder_;
    ad::nn::FullyConnParams decoder_;

    size_t input_size_;
    size_t output_size_;

    ad::Var ComputeModel(
            ad::ComputationGraph& g,
            const std::vector<WordFeatures>& ws) const;
  public:
    SequenceClassifier(
        size_t out_sz,
        size_t hidden_size,
        size_t wordvec_size,
        size_t vocab_size);

    std::string Serialize() const;
    static SequenceClassifier FromSerialized(std::istream& file);

    Eigen::MatrixXd ComputeClass(const std::vector<WordFeatures>& ws) const;

    double Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

