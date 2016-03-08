#pragma once

#include <ad/ad.h>

struct HierarchicalCharRNNParams {
    std::shared_ptr<ad::Param> start_;
    ad::nn::Hashtable chars_emb_;
    ad::nn::GRUParams fwd_;
    ad::nn::GRUParams bwd_;
    ad::nn::GRUParams dec_;
    ad::nn::GRUParams chars_;
    ad::nn::FullyConnParams out_;
    ad::nn::FullyConnParams w_to_chars_;
    ad::nn::FullyConnParams chars_to_w_;
    size_t vocab_size_;

    public:
    HierarchicalCharRNNParams(size_t nb_chars, size_t whidden_size, size_t chidden_size)
        : start_(std::make_shared<ad::Param>(chidden_size, 1, ad::Gaussian(0, 0.1))),
        chars_emb_(10, nb_chars),
        fwd_(whidden_size, whidden_size),
        bwd_(whidden_size, whidden_size),
        dec_(whidden_size, 2 * whidden_size + chidden_size),
        chars_(chidden_size, 10),
        out_(nb_chars, chidden_size),
        w_to_chars_(chidden_size, whidden_size),
        chars_to_w_(whidden_size, chidden_size),
        vocab_size_(nb_chars) {
    }
};

class AutoEncoderEncoder {
    ad::nn::Hashtable chars_emb_;
    ad::nn::GRULayer fwd_;
    ad::nn::GRULayer bwd_;
    ad::nn::GRULayer chars_;
    ad::nn::FullyConnLayer chars_to_w_;
    size_t vocab_size_;
    int space_;

    ad::Var char_init_;

    public:
    std::vector<ad::Var> words_embeddings_;

    AutoEncoderEncoder(ad::ComputationGraph& g, const HierarchicalCharRNNParams& params, int space)
        : chars_emb_(params.chars_emb_),
        fwd_(g, params.fwd_),
        bwd_(g, params.bwd_),
        chars_(g, params.chars_),
        chars_to_w_(g, params.chars_to_w_),
        vocab_size_(params.vocab_size_),
        space_(space),
        char_init_(chars_.GetHidden()) {
    }

    ad::Var Step(ad::ComputationGraph& g, const std::vector<WordFeatures>& ws) {
        ad::Var sentence_embedding(nullptr);
        ad::Var word_embedding(nullptr);

        size_t i = 0;
        while (ws[ws.size() - 1 - i].idx == space_) {
            ++i;
        }
        for (; i < ws.size(); ++i) {
            auto& wf = ws[ws.size() - 1 - i];
            if (wf.idx == space_) {
                // project word onto sentence space
                word_embedding = Tanh(chars_to_w_.Compute(word_embedding));

                // save the word embedding for ladder loss
                words_embeddings_.push_back(word_embedding);

                // reinit char rnn
                chars_.SetHidden(char_init_);
            } else {
                // append char
                word_embedding = chars_.Step(chars_emb_.MakeVarFor(g, wf.idx));
            }
        }

        if (ws[0].idx != space_) {
            // project word onto sentence space
            word_embedding = Tanh(chars_to_w_.Compute(word_embedding));

            // save the word embedding for ladder loss
            words_embeddings_.push_back(word_embedding);
        }

        ad::Var fwd_embed(nullptr);
        ad::Var bwd_embed(nullptr);
        for (int i = 0; i < words_embeddings_.size(); ++i) {
            fwd_embed = fwd_.Step(words_embeddings_[i]);
            bwd_embed = bwd_.Step(words_embeddings_[words_embeddings_.size() - i - 1]);
        }

        // No attention

        return ad::ColAppend(fwd_embed, bwd_embed);
    }
};

class AutoEncoderDecoder {
    ad::nn::Hashtable chars_emb_;
    ad::nn::GRULayer dec_;
    ad::nn::GRULayer chars_;
    ad::nn::FullyConnLayer out_;
    ad::nn::FullyConnLayer w_to_chars_;
    ad::Var start_;
    ad::Var last_word_;

    ad::Var embedded_sentence_;

    size_t vocab_size_;
    int space_;

    public:

    AutoEncoderDecoder(
            ad::ComputationGraph& g,
            const HierarchicalCharRNNParams& params,
            int space)
        : chars_emb_(params.chars_emb_),
        start_(g.CreateParam(params.start_)),
        dec_(g, params.dec_),
        chars_(g, params.chars_),
        out_(g, params.out_),
        w_to_chars_(g, params.w_to_chars_),
        last_word_(nullptr),
        embedded_sentence_(nullptr),
        vocab_size_(params.vocab_size_),
        space_(space) {
    }

    void SetEmbedding(ad::Var embedded_sentence) {
        embedded_sentence_ = embedded_sentence;
        last_word_ = Tanh(w_to_chars_.Compute(dec_.Step(ad::ColAppend(embedded_sentence_, start_))));
        chars_.SetHidden(last_word_);
    }

    ad::Var Step(ad::ComputationGraph& g, int c) {
        std::vector<ad::Var> chars_out;

        if (c == space_) {
            last_word_ = Tanh(w_to_chars_.Compute(dec_.Step(ad::ColAppend(embedded_sentence_, last_word_))));
            chars_.SetHidden(last_word_);

            return out_.Compute(chars_.Step(chars_emb_.MakeVarFor(g, 0)));
        }

        ad::Var prob_char = out_.Compute(chars_.Step(chars_emb_.MakeVarFor(g, c)));
        return prob_char;
    }

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

        return J;
    }
};

struct AutoEncoderFullParams {
    HierarchicalCharRNNParams enc_;
    HierarchicalCharRNNParams dec_;

    size_t vocab_size_;

    AutoEncoderFullParams(size_t nb_chars, size_t whidden_size, size_t chidden_size)
        : enc_(nb_chars, whidden_size, chidden_size),
        dec_(nb_chars, whidden_size, chidden_size),
        vocab_size_(nb_chars) {
    }
};

struct AutoEncoderFull {
    AutoEncoderEncoder encoder_;
    AutoEncoderDecoder decoder_;
    size_t vocab_size_;
    size_t space_;

    AutoEncoderFull(
            ad::ComputationGraph& g,
            const AutoEncoderFullParams& params,
            int space)
        : encoder_(g, params.enc_, space),
        decoder_(g, params.dec_, space),
        vocab_size_(params.vocab_size_),
        space_(space) {
    }

    std::vector<ad::Var> Step(
            ad::ComputationGraph& g,
            const std::vector<WordFeatures>& ws) {
        return SupervisedStep(g, ws);
    }

    std::vector<ad::Var> SupervisedStep(
            ad::ComputationGraph& g,
            const std::vector<WordFeatures>& ws) {
        ad::Var embedding = encoder_.Step(g, ws);
        decoder_.SetEmbedding(embedding);
        std::vector<ad::Var> chars;
        chars.push_back(decoder_.Step(g, 0));
        for (auto& wf : ws) {
            chars.push_back(decoder_.Step(g, wf.idx));
        }
        return chars;
    }

    std::vector<ad::Var> FreestyleStep(
            ad::ComputationGraph& g,
            const std::vector<WordFeatures>& ws) {
        ad::Var embedding = encoder_.Step(g, ws);
        decoder_.SetEmbedding(embedding);
        std::vector<ad::Var> chars;

        chars.push_back(decoder_.Step(g, 0));
        for (size_t i = 0; i < ws.size(); ++i) {
            int prev = ad::utils::OneHotVectorDecode(chars.back().value());
            chars.push_back(decoder_.Step(g, prev));
        }
        return chars;
    }

    ad::Var Cost(
            ad::ComputationGraph& g,
            const std::vector<ad::Var>& dec,
            const std::vector<WordFeatures>& original) {

        Eigen::MatrixXd zero(1, 1);
        zero << 0;
        ad::Var J = g.CreateParam(zero);
        for (size_t i = 0; i < original.size(); ++i) {
            ad::Var target = g.CreateParam(
                    ad::utils::OneHotColumnVector(original[i].idx, vocab_size_));

            J = J + ad::Sum(ad::SoftmaxLoss(dec[i], target));
        }

        //J = (1.0 / dec.size()) * J;

#if 0
        size_t nb_words = std::min(enc.encoder_.words_embeddings_.size(),
                decoder_.words_embeddings_.size());

        for (size_t i = 0; i < nb_words; ++i) {
            J = J + 100 * ad::SSE(enc.encoder_.words_embeddings_[i],
                    decoder_.words_embeddings_[i]);
        }
#endif
        return J;
    }
};
