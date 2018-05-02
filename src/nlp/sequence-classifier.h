#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <ad/ad.h>

#include "document.h"

struct SequenceClassifier {
    ad::nn::Hashtable words;
    ad::nn::RNNParams encoder;
    ad::nn::FullyConnParams decoder;

    size_t input_size_;
    size_t output_size_;

    SequenceClassifier(
        size_t out_sz,
        size_t hidden_size,
        size_t wordvec_size,
        size_t vocab_size);

    std::string Serialize() const;
    static SequenceClassifier FromSerialized(std::istream& file);

    ad::Matrix& ComputeClass(const std::vector<WordFeatures>& ws) const;

    double Train(const Document& doc);
    double Test(const Document& doc);

    void ResizeInput(size_t in);
};

class SequenceClassifierGraph {
    private:
        ad::nn::Hashtable words_;
        ad::nn::RNNLayer encoder_;
        ad::nn::FullyConnLayer decoder_;

        size_t input_size_;
        size_t output_size_;

    public:
        SequenceClassifierGraph(
                ad::ComputationGraph& g,
                const SequenceClassifier& params,
                size_t input_size,
                size_t output_size)
            : words_(params.words),
            encoder_(g, params.encoder),
            decoder_(g, params.decoder),
            input_size_(input_size),
            output_size_(output_size) {
        }

        ad::Var Step(ad::ComputationGraph& g,
                const std::vector<WordFeatures>& ws);

        ad::Var Cost(ad::ComputationGraph& g, ad::Var h, int k) {
            ad::Var y = g.CreateParam(
                    ad::utils::OneHotColumnVector(k, output_size_));
            return ad::Sum(ad::SoftmaxLoss(h, y));
        }
};
