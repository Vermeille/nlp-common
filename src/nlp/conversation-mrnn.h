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
            const std::vector<ad::Var>& pred,
            const std::vector<WordFeatures>& exp) {

        ad::RWMatrix zero(1, 1);
        zero(0, 0) = 0;
        ad::Var J = g.CreateParam(ad::Matrix(zero));
        for (size_t i = 1; i < exp.size(); ++i) {
            ad::Var target = g.CreateParam(
                    ad::utils::OneHotColumnVector(exp[i].idx, vocab_size_));

            J = J + Sum(SoftmaxLoss(pred[i - 1], target));
        }

        return J;
    }
};


