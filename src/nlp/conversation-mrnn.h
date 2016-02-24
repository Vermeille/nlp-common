#pragma once

#include <ad/ad.h>

struct ConversationParams {
    ad::nn::DiscreteMRNNParams encoder_;
    ad::nn::FullyConnParams fc_;

    size_t vocab_size_;
    size_t hidden_size_;

    public:
    ConversationParams(size_t vocab_size, size_t hidden_size)
        : encoder_(hidden_size, vocab_size),
        fc_(vocab_size, hidden_size),
        vocab_size_(vocab_size),
        hidden_size_(hidden_size) {
    }

};

class Conversation {
    ad::nn::DiscreteMRNNLayer encoder_;
    ad::nn::FullyConnLayer fc_;

    size_t vocab_size_;
    size_t hidden_size_;

    public:
    Conversation(ad::ComputationGraph& g, const ConversationParams& params)
        : encoder_(g, params.encoder_),
        fc_(g, params.fc_),
        vocab_size_(params.vocab_size_),
        hidden_size_(params.hidden_size_) {
    }

    ad::Var Step(ad::ComputationGraph& g, int wf) {
        return fc_.Compute(encoder_.Step(wf));
    }

    ad::Var Step(ad::ComputationGraph& g, const WordFeatures& wf) {
        return Step(g, wf.idx);
    }

    ad::Var Cost(ad::ComputationGraph& g,
            ad::Var pred,
            const WordFeatures& exp) {

        ad::Var target = g.CreateParam(
                ad::utils::OneHotColumnVector(exp.idx, vocab_size_));

        return Sum(SoftmaxLoss(pred, target));
    }
};


