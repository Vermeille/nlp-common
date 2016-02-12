#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>

#include <ad/ad.h>

#include "document.h"

class SequenceTagger {
    ad::nn::Hashtable words_;
    ad::nn::RNNLayerParams rnn_;
    ad::nn::FullyConnParams fc_;

    size_t output_size_;

    std::vector<ad::Var> ComputeModel(
            ad::ComputationGraph& g,
            std::vector<WordFeatures>::const_iterator begin,
            std::vector<WordFeatures>::const_iterator end) const;

  public:
    SequenceTagger(size_t in_sz, size_t out_sz);

    SequenceTagger();

    std::string Serialize() const;

    static SequenceTagger FromSerialized(std::istream& file);

    Label ComputeTagForWord(
            const WordFeatures& ws,
            const WordFeatures& prev,
            double* probabilities) const;

    void Compute(std::vector<WordFeatures>& ws);

    int Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

