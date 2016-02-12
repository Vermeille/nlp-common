#include <glog/logging.h>

#include "sequence-tagger.h"

static const int wordvec_size = 10;
static const int hidden_size = 30;

SequenceTagger::SequenceTagger(size_t vocab_sz, size_t out_sz)
        : words_(wordvec_size, vocab_sz),
        rnn_(hidden_size, wordvec_size),
        fc_(out_sz, hidden_size),
        output_size_(out_sz) {
}

SequenceTagger::SequenceTagger()
        : words_(0, wordvec_size),
        rnn_(hidden_size, wordvec_size),
        fc_(0, hidden_size),
        output_size_(0) {
}

void SequenceTagger::Compute(std::vector<WordFeatures>& ws) {
    ad::ComputationGraph g;

    const auto& outs = ComputeModel(g, ws.begin(), ws.end());
    for (size_t i = 0; i < outs.size(); ++i) {
        ws[i].pos = ad::utils::OneHotVectorDecode(outs[i].value());
    }
}

std::vector<ad::Var> SequenceTagger::ComputeModel(
        ad::ComputationGraph& g,
        std::vector<WordFeatures>::const_iterator begin,
        std::vector<WordFeatures>::const_iterator end) const {
    using namespace ad;

    nn::RNNLayer rnn(g, rnn_);
    nn::FullyConnLayer fc(g, fc_);

    std::vector<Var> out;
    for (; begin != end; ++begin) {
        Var tag =
            Softmax(fc.Compute(rnn.Step(words_.MakeVarFor(g, begin->idx))));
        out.push_back(tag);
    }
    return out;
}

int SequenceTagger::Train(const Document& doc) {
    using namespace ad;
    double nll = 0;
    int nb_correct = 0;
    int nb_tokens = 0;

    int cur = 0;
    for (auto& ex : doc.examples) {
        auto begin = ex.inputs.begin();
        auto end = begin + 1;
        for (size_t i = 0; i < ex.inputs.size(); ++i) {
            auto& wf = ex.inputs[i];

            ComputationGraph g;

            Eigen::MatrixXd yt_mat
                = ad::utils::OneHotColumnVector(wf.pos, output_size_);
            Var yt = g.CreateParam(yt_mat);

            auto outs = ComputeModel(g, begin, end);

            Var Jt = MSE(outs.back(), yt);// + 1e-4 * nn::L2ForAllParams(outs);

            nll += Jt.value().sum();

            g.BackpropFrom(Jt, 5);

            opt::SGD sgd(0.01);
            g.Update(sgd);
            ++end;

            Label predicted = ad::utils::OneHotVectorDecode(outs.back().value());

            nb_correct += predicted == ex.inputs[i].pos ? 1 : 0;
            ++nb_tokens;
        }
        ++cur;
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string SequenceTagger::Serialize() const {
    std::ostringstream out;
    out << words_.Serialize();
    rnn_.Serialize(out);
    fc_.Serialize(out);
    return out.str();
}

SequenceTagger SequenceTagger::FromSerialized(std::istream& in) {
    SequenceTagger seq(0, 0);

    seq.words_ = ad::nn::Hashtable::FromSerialized(in);
    seq.rnn_ = ad::nn::RNNLayerParams::FromSerialized(in);
    seq.fc_ = ad::nn::FullyConnParams::FromSerialized(in);
    return seq;
}

void SequenceTagger::ResizeInput(size_t in) {
    words_.ResizeVocab(in);
}

void SequenceTagger::ResizeOutput(size_t out) {
    rnn_.ResizeOutput(out);
    output_size_ = (out > output_size_) ? out : output_size_;
}

