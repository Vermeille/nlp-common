#include <glog/logging.h>

#include "sequence-tagger.h"

static const int wordvec_size = 10;
static const int hidden_size = 30;

SequenceTagger::SequenceTagger(size_t vocab_sz, size_t out_sz)
        : words_(vocab_sz, wordvec_size),
        rnn_(out_sz, wordvec_size, hidden_size),
        output_size_(out_sz) {
}

SequenceTagger::SequenceTagger()
        : words_(0, wordvec_size),
        rnn_(0, wordvec_size, hidden_size),
        output_size_(0) {
}

void SequenceTagger::Compute(std::vector<WordFeatures>& ws) {
    std::vector<ad::Var> woxes;

    ad::ComputationGraph g;

    const auto& outs = ComputeModel(g, woxes, ws.begin(), ws.end());
    for (size_t i = 0; i < outs.out.first.size(); ++i) {
        ws[i].pos = ad::utils::OneHotVectorDecode(outs.out.first[i].value());
    }
}

ad::nn::NeuralOutput<std::pair<std::vector<ad::Var>, ad::Var>>
SequenceTagger::ComputeModel(
        ad::ComputationGraph& g,
        std::vector<ad::Var>& woxes,
        std::vector<WordFeatures>::const_iterator begin,
        std::vector<WordFeatures>::const_iterator end) const {
    using namespace ad;

    while (begin != end) {
        woxes.push_back(words_.MakeVarFor(g, begin->idx));
        ++begin;
    }
    auto in = nn::InputLayer(woxes);
    auto out = rnn_.Compute(in);
    for (size_t i = 0; i < out.out.first.size(); ++i) {
        out.out.first[i] = Softmax(out.out.first[i]);
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
            std::vector<Var> woxes;

            ComputationGraph g;

            Eigen::MatrixXd yt_mat
                = ad::utils::OneHotColumnVector(wf.pos, output_size_);
            Var yt = g.CreateParam(yt_mat);

            auto outs = ComputeModel(g, woxes, begin, end);

            Var Jt = MSE(outs.out.first.back(), yt);// + 1e-4 * nn::L2ForAllParams(outs);

            nll += Jt.value().sum();

            g.BackpropFrom(Jt);

            opt::SGD sgd(0.01);
            g.Update(sgd, *outs.params);
            for (auto w : woxes) {
                g.Update(sgd, {w});
            }
            ++end;

            Label predicted = ad::utils::OneHotVectorDecode(outs.out.first.back().value());

            nb_correct += predicted == ex.inputs[i].pos ? 1 : 0;
            ++nb_tokens;
        }
        ++cur;
        std::cerr << cur << "      " << nll << "               \r";
        //std::cerr.flush();
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string SequenceTagger::Serialize() const {
    std::ostringstream out;
    out << words_.Serialize();
    rnn_.Serialize(out);
    return out.str();
}

SequenceTagger SequenceTagger::FromSerialized(std::istream& in) {
    SequenceTagger seq(0, 0);

    seq.words_ = ad::nn::Hashtable::FromSerialized(in);
    seq.rnn_ = ad::nn::RNNLayer1::FromSerialized(in);
    return seq;
}

void SequenceTagger::ResizeInput(size_t in) {
    words_.ResizeVocab(in);
}

void SequenceTagger::ResizeOutput(size_t out) {
    rnn_.ResizeOutput(out);
    output_size_ = (out > output_size_) ? out : output_size_;
}

