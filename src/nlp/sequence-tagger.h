#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>

#include <ad/ad.h>

#include "document.h"

class SequenceTagger {
    // very simple RNN model
    // z(t) = Wox * x(t) + Woo * h(t - 1) + b
    // h(t) = ReLu(z(t))
    // o(t) = Softmax(z(t))
    std::vector<std::shared_ptr<Eigen::MatrixXd>> wox_; //(ouput_sz, 1)[input]
    std::shared_ptr<Eigen::MatrixXd> woo_; //(ouput_sz, output_sz)
    std::shared_ptr<Eigen::MatrixXd> b_; //(out_sz, 1)

    size_t input_size_;
    size_t output_size_;

    std::vector<ad::Var> ComputeModel(
            ad::ComputationGraph& g,
            std::vector<ad::Var>& woxes, ad::Var woo, ad::Var b,
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

