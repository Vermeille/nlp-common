#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>

#include <ad/ad.h>

#include "document.h"

class SequenceTagger {
    std::vector<std::shared_ptr<Eigen::MatrixXd>> wox_; //(ouput_sz, 1)[input]
    ad::nn::RNNLayer1 rnn_;

    size_t vocab_size_;
    size_t output_size_;

    ad::nn::NeuralOutput<std::pair<std::vector<ad::Var>, ad::Var>>
    ComputeModel(
            ad::ComputationGraph& g,
            std::vector<ad::Var>& woxes,
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

