#pragma once

#include <ad/ad.h>

struct ConversationParams {
    std::shared_ptr<ad::Param> start_;
    //ad::nn::LSTMParams words_;
    ad::nn::RNNParams words_;
    ad::nn::DiscreteMRNNParams chars_;
    ad::nn::FullyConnParams out_;
    ad::nn::FullyConnParams w_to_chars_;
    ad::nn::FullyConnParams chars_to_w_;
    size_t vocab_size_;

    public:
    ConversationParams(size_t nb_chars, size_t whidden_size, size_t chidden_size)
        : start_(std::make_shared<ad::Param>(whidden_size, 1, ad::Gaussian(0, 0.1))),
        words_(whidden_size, whidden_size),
        chars_(chidden_size, nb_chars),
        out_(nb_chars, chidden_size),
        w_to_chars_(chidden_size, whidden_size),
        chars_to_w_(whidden_size, chidden_size),
        vocab_size_(nb_chars) {
    }
};

class Conversation {
    ad::Var start_;
    //ad::nn::LSTMLayer words_;
    ad::nn::RNNLayer words_;
    ad::nn::DiscreteMRNNLayer chars_;
    ad::nn::FullyConnLayer out_;
    ad::Var last_char_hidden_;
    ad::nn::FullyConnLayer w_to_chars_;
    ad::nn::FullyConnLayer chars_to_w_;

    size_t vocab_size_;
    int space_;

    public:
    Conversation(ad::ComputationGraph& g, const ConversationParams& params, int space)
        : start_(g.CreateParam(params.start_)),
        words_(g, params.words_),
        chars_(g, params.chars_),
        out_(g, params.out_),
        w_to_chars_(g, params.w_to_chars_),
        chars_to_w_(g, params.chars_to_w_),
        last_char_hidden_(nullptr),
        vocab_size_(params.vocab_size_),
        space_(space) {
    }

    ad::Var Step(ad::ComputationGraph& g, int wf) {
        if (wf == 0) {
            chars_.SetHidden(Tanh(w_to_chars_.Compute(words_.Step(start_))));
            last_char_hidden_ = chars_.Step(wf);
            return out_.Compute(last_char_hidden_);
        } else if (wf == space_) {
            chars_.SetHidden(
                    Tanh(w_to_chars_.Compute(
                        words_.Step(
                            chars_to_w_.Compute(
                                last_char_hidden_)))));
            last_char_hidden_ = chars_.Step(0);
            return out_.Compute(last_char_hidden_);
        }
        last_char_hidden_ = chars_.Step(wf);
        return out_.Compute(last_char_hidden_);
    }

    ad::Var Step(ad::ComputationGraph& g, const WordFeatures& wf) {
        return Step(g, wf.idx);
    }

#if 0
    ad::Var Cost(ad::ComputationGraph& g,
            ad::Var pred,
            const WordFeatures& exp) {

        ad::Var target = g.CreateParam(
                ad::utils::OneHotColumnVector(exp.idx, vocab_size_));

        return Sum(SoftmaxLoss(pred, target));
    }
#else
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
#endif
};


