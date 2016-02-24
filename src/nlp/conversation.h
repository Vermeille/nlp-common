#pragma once

#include <ad/ad.h>

struct ConversationParams {
    ad::nn::LSTMParams encoder_;
    ad::nn::LSTMParams encoder2_;
    ad::nn::FullyConnParams fc_;
    ad::opt::Minibatch<ad::opt::Adagrad> adagrad_;

    size_t vocab_size_;
    size_t hidden_size_;

    public:
    ConversationParams(size_t vocab_size, size_t hidden_size)
        : encoder_(hidden_size, 5),
        encoder2_(hidden_size, hidden_size),
        fc_(vocab_size, hidden_size),
        adagrad_(100, ad::opt::Adagrad()),
        vocab_size_(vocab_size),
        hidden_size_(hidden_size) {
    }
};

class Conversation {
    private:
        ad::nn::Hashtable char_embed_;
        ad::nn::LSTMLayer encoder_;
        ad::nn::LSTMLayer encoder2_;
        ad::nn::FullyConnLayer fc_;

        size_t vocab_size_;

    public:
        Conversation(ad::ComputationGraph& g, const ConversationParams& params)
            : char_embed_(5, params.vocab_size_),
            encoder_(g, params.encoder_),
            encoder2_(g, params.encoder2_),
            fc_(g, params.fc_),
            vocab_size_(params.vocab_size_) {
            }


        ad::Var Step(ad::ComputationGraph& g, int wf) {
            return fc_.Compute(encoder2_.Step(encoder_.Step(char_embed_.MakeVarFor(g, wf))));
        }

        ad::Var Step(ad::ComputationGraph& g, const WordFeatures&wf) {
            return Step(g, wf.idx);
        }

#if 0
        ad::Var Cost(ad::ComputationGraph& g,
                const std::vector<ad::Var>& pred,
                const std::vector<WordFeatures>& exp) {

            Eigen::MatrixXd zero(1, 1);
            zero << 0;
            ad::Var J = g.CreateParam(zero);
            for (size_t i = 1; i < exp.size(); ++i) {
                ad::Var target = g.CreateParam(
                        ad::utils::OneHotColumnVector(exp[i].idx, vocab_size_));

                J = J + Sum(SoftmaxLoss(pred[i - 1], target));
            }

            return J + 1e-6 * (ad::nn::L2(fc_.Params())
                    + ad::nn::L2(encoder_.Params())
                    + ad::nn::L2(encoder2_.Params()));
        }
#else
    ad::Var Cost(ad::ComputationGraph& g,
            ad::Var pred,
            const WordFeatures& exp) {

        ad::Var target = g.CreateParam(
                ad::utils::OneHotColumnVector(exp.idx, vocab_size_));

        return Sum(SoftmaxLoss(pred, target));

        //return /*(1.0 / pred.size()) */ J;
            //+ 1e-6 * (ad::nn::L2(fc_.Params()) + ad::nn::L2(encoder_.Params()));
    }
#endif
};

