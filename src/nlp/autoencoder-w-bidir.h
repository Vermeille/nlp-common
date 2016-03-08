#pragma once

#include <ad/ad.h>

struct HierarchicalCharRNNParams {
    ad::nn::Hashtable words_emb_;
    ad::nn::GRUParams fwd_;
    ad::nn::GRUParams bwd_;
    ad::nn::GRUParams dec_;
    ad::nn::FullyConnParams out_;
    size_t vocab_size_;

    public:
    HierarchicalCharRNNParams(size_t nb_chars, size_t whidden_size, size_t chidden_size)
        :
        words_emb_(chidden_size, nb_chars),
        fwd_(whidden_size, chidden_size),
        bwd_(whidden_size, chidden_size),
        dec_(whidden_size, 2 * whidden_size + chidden_size),
        out_(nb_chars, whidden_size),
        vocab_size_(nb_chars) {
    }
};

class AutoEncoderEncoder {
    ad::nn::Hashtable words_emb_;
    ad::nn::GRULayer fwd_;
    ad::nn::GRULayer bwd_;
    size_t vocab_size_;

    public:
    std::vector<ad::Var> words_embeddings_;

    AutoEncoderEncoder(ad::ComputationGraph& g, const HierarchicalCharRNNParams& params, int space)
        : words_emb_(params.words_emb_),
        fwd_(g, params.fwd_),
        bwd_(g, params.bwd_),
        vocab_size_(params.vocab_size_) {
    }

    ad::Var Step(ad::ComputationGraph& g, const std::vector<WordFeatures>& ws) {
        ad::Var fwd_embed(nullptr);
        ad::Var bwd_embed(nullptr);
        for (int i = 0; i < ws.size(); ++i) {
            fwd_embed = fwd_.Step(words_emb_.MakeVarFor(g, ws[i].idx));
            bwd_embed = bwd_.Step(words_emb_.MakeVarFor(g, ws[ws.size() - i - 1].idx));
        }

        // No attention

        return ad::ColAppend(fwd_embed, bwd_embed);
    }
};

class AutoEncoderDecoder {
    ad::nn::Hashtable words_emb_;
    ad::nn::GRULayer dec_;
    ad::nn::FullyConnLayer out_;

    ad::Var embedded_sentence_;

    size_t vocab_size_;

    public:

    AutoEncoderDecoder(
            ad::ComputationGraph& g,
            const HierarchicalCharRNNParams& params,
            int space)
        : words_emb_(params.words_emb_),
        dec_(g, params.dec_),
        out_(g, params.out_),
        embedded_sentence_(nullptr),
        vocab_size_(params.vocab_size_) {
    }

    void SetEmbedding(ad::Var embedded_sentence) {
        embedded_sentence_ = embedded_sentence;
    }

    ad::Var Step(ad::ComputationGraph& g, int w) {
        return
            out_.Compute(
                Tanh(dec_.Step(
                    ad::ColAppend(
                        embedded_sentence_,
                        words_emb_.MakeVarFor(g, w)))));
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
        for (size_t i = 0; i < 80; ++i) {
            int prev = ad::utils::OneHotVectorDecode(chars.back().value());
            if (prev == 0) {
                break;
            }
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

        return J + ad::Sum(ad::SoftmaxLoss(dec.back(),
                    g.CreateParam(
                        ad::utils::OneHotColumnVector(0, vocab_size_))));
    }
};
