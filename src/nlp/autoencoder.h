#pragma once

#include <ad/ad.h>

struct HierarchicalCharRNNParams {
    std::shared_ptr<ad::Param> start_;
    ad::nn::Hashtable chars_emb_;
    ad::nn::GRUParams words1_;
    ad::nn::GRUParams words2_;
    ad::nn::GRUParams words3_;
    ad::nn::GRUParams words4_;
    ad::nn::LSTMParams chars_;
    ad::nn::FullyConnParams out_;
    ad::nn::FullyConnParams w_to_chars_;
    ad::nn::FullyConnParams chars_to_w_;
    size_t vocab_size_;

    public:
    HierarchicalCharRNNParams(size_t nb_chars, size_t whidden_size, size_t chidden_size)
        : start_(std::make_shared<ad::Param>(whidden_size, 1, ad::Gaussian(0, 0.1))),
        chars_emb_(10, nb_chars),
        words1_(whidden_size, whidden_size),
        words2_(whidden_size, whidden_size),
        words3_(whidden_size, whidden_size),
        words4_(whidden_size, whidden_size),
        chars_(chidden_size, 10),
        out_(nb_chars, chidden_size),
        w_to_chars_(chidden_size, whidden_size),
        chars_to_w_(whidden_size, chidden_size),
        vocab_size_(nb_chars) {
    }
};

class AutoEncoderEncoder {
    ad::nn::Hashtable chars_emb_;
    ad::nn::GRULayer words1_;
    ad::nn::GRULayer words2_;
    ad::nn::GRULayer words3_;
    ad::nn::GRULayer words4_;
    ad::nn::LSTMLayer chars_;
    ad::nn::FullyConnLayer chars_to_w_;
    size_t vocab_size_;
    int space_;

    ad::Var char_init_;

    public:
    std::vector<ad::Var> words_embeddings_;

    AutoEncoderEncoder(ad::ComputationGraph& g, const HierarchicalCharRNNParams& params, int space)
        : chars_emb_(params.chars_emb_),
        words1_(g, params.words1_),
        words2_(g, params.words2_),
        words3_(g, params.words3_),
        words4_(g, params.words4_),
        chars_(g, params.chars_),
        chars_to_w_(g, params.chars_to_w_),
        vocab_size_(params.vocab_size_),
        space_(space),
        char_init_(chars_.GetHidden()) {
    }

    ad::Var AppendAllHiddens(ad::Var in) {
        ad::Var h1 = words1_.Step(in);
        ad::Var h2 = words2_.Step(h1);
        ad::Var h3 = words3_.Step(h2);
        ad::Var h4 = words4_.Step(h3);
        return ad::ColAppend(h1, ad::ColAppend(h2, ad::ColAppend(h3, h4)));
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

                // add the word in the sentence
                sentence_embedding = AppendAllHiddens(word_embedding);

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

            // add the word in the sentence
            sentence_embedding = AppendAllHiddens(word_embedding);
        }

        std::reverse(words_embeddings_.begin(), words_embeddings_.end());
        return sentence_embedding;
    }
};

class AutoEncoderDecoder {
    ad::nn::Hashtable chars_emb_;
    ad::nn::GRULayer words1_;
    ad::nn::GRULayer words2_;
    ad::nn::GRULayer words3_;
    ad::nn::GRULayer words4_;
    ad::nn::LSTMLayer chars_;
    ad::nn::FullyConnLayer out_;
    ad::nn::FullyConnLayer w_to_chars_;
    ad::Var start_;
    ad::Var last_word_;

    size_t vocab_size_;
    int space_;

    public:
    std::vector<ad::Var> words_embeddings_; // for ladder net

    AutoEncoderDecoder(
            ad::ComputationGraph& g,
            const HierarchicalCharRNNParams& params,
            int space)
        : chars_emb_(params.chars_emb_),
        start_(g.CreateParam(params.start_)),
        words1_(g, params.words1_),
        words2_(g, params.words2_),
        words3_(g, params.words3_),
        words4_(g, params.words4_),
        chars_(g, params.chars_),
        out_(g, params.out_),
        w_to_chars_(g, params.w_to_chars_),
        last_word_(nullptr),
        vocab_size_(params.vocab_size_),
        space_(space) {
    }

    void SetEmbedding(ad::Var embedded_sentence) {
        int split = embedded_sentence.value().rows() / 4;
        ad::Var h1 = ad::ColSplit(embedded_sentence, 0, split);
        ad::Var h2 = ad::ColSplit(embedded_sentence, split, split);
        ad::Var h3 = ad::ColSplit(embedded_sentence, 2 * split, split);
        ad::Var h4 = ad::ColSplit(embedded_sentence, 3 * split, split);
        words1_.SetHidden(h1);
        words2_.SetHidden(h2);
        words3_.SetHidden(h3);
        words4_.SetHidden(h4);
        last_word_ = words4_.Step(words3_.Step(words2_.Step(words1_.Step(start_))));
        ad::Var w_chars = Tanh(w_to_chars_.Compute(last_word_));
        words_embeddings_.push_back(last_word_);
        chars_.SetHidden(w_chars);
    }

    ad::Var Step(ad::ComputationGraph& g, int c) {
        std::vector<ad::Var> chars_out;

        if (c == space_) {
            last_word_ = words4_.Step(words3_.Step(words2_.Step(words1_.Step(g.CreateParam(last_word_.value())))));
            ad::Var w_chars = Tanh(w_to_chars_.Compute(last_word_));
            words_embeddings_.push_back(last_word_);
            chars_.SetHidden(w_chars);

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
