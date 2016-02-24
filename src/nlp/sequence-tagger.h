#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <Eigen/Dense>

#include <ad/ad.h>

#include "document.h"

struct SequenceTaggerParams {
    ad::nn::Hashtable words;
    ad::nn::RNNLayerParams rnn;
    ad::nn::FullyConnParams fc;

    size_t output_size_;

    SequenceTaggerParams(size_t in_sz, size_t out_sz);

    std::string Serialize() const;

    static SequenceTaggerParams FromSerialized(std::istream& file);

    void Compute(std::vector<WordFeatures>& ws);

    double Test(const Document& doc);
    double Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

class SequenceTagger {
    ad::nn::Hashtable words_;
    ad::nn::RNNLayer rnn_;
    ad::nn::FullyConnLayer fc_;
    size_t output_size_;

    public:
        SequenceTagger(ad::ComputationGraph& g,
                size_t output_size,
                const SequenceTaggerParams& params,
                bool learnable = true)
            : words_(params.words),
            rnn_(g, params.rnn, learnable),
            fc_(g, params.fc, learnable),
            output_size_(output_size) {
        }

        ad::Var Step(ad::ComputationGraph& g, const WordFeatures& wf) {
            return Softmax(fc_.Compute(rnn_.Step(words_.MakeVarFor(g, wf.idx))));
        }

        ad::Var Cost(ad::ComputationGraph& g, ad::Var h, const WordFeatures& wf) {
            ad::Var yt = g.CreateParam(
                ad::utils::OneHotColumnVector(wf.pos, output_size_));
            return ad::MSE(h, yt);// + 1e-4 * nn::L2ForAllParams(outs);
        }
};
