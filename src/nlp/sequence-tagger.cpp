#include <glog/logging.h>

#include "sequence-tagger.h"

static const int wordvec_size = 10;
static const int hidden_size = 30;

SequenceTaggerParams::SequenceTaggerParams(size_t vocab_sz, size_t out_sz)
        : words(wordvec_size, vocab_sz),
        rnn(hidden_size, wordvec_size),
        fc(out_sz, hidden_size),
        output_size_(out_sz) {
}

void SequenceTaggerParams::Compute(std::vector<WordFeatures>& ws) {
    ad::ComputationGraph g;

    SequenceTagger tagger(g, output_size_, *this);
    for (size_t i = 0; i < ws.size(); ++i) {
        ws[i].pos = ad::utils::OneHotVectorDecode(
                Softmax(tagger.Step(g, ws[i])).value());
    }
}

double SequenceTaggerParams::Test(const Document& doc) {
    using namespace ad;
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        ComputationGraph g;
        SequenceTagger tagger(g, output_size_, *this);
        for (size_t i = 0; i < ex.inputs.size(); ++i) {
            auto& wf = ex.inputs[i];
            ad::Var out = Softmax(tagger.Step(g, wf));

            Label predicted = ad::utils::OneHotVectorDecode(out.value());

            nb_correct += predicted == wf.pos ? 1 : 0;
            ++nb_tokens;
        }
    }
    return  nb_correct * 100 / nb_tokens;
}

double SequenceTaggerParams::Train(const Document& doc) {
    using namespace ad;

    train::WholeSequenceTaggerTrainer trainer(new ad::opt::Adagrad(0.1));
    for (int i = 0; i < 3; ++i) {
        for (auto& ex : doc.examples) {
            trainer.Step(ex.inputs, ex.inputs,
                    [&](ad::ComputationGraph& g) {
                    return SequenceTagger(g, output_size_, *this);
                    });
        }
    }
    return Test(doc);
}

std::string SequenceTaggerParams::Serialize() const {
    std::ostringstream out;
    out << words.Serialize();
    rnn.Serialize(out);
    fc.Serialize(out);
    return out.str();
}

SequenceTaggerParams SequenceTaggerParams::FromSerialized(std::istream& in) {
    SequenceTaggerParams seq(0, 0);

    seq.words = ad::nn::Hashtable::FromSerialized(in);
    seq.rnn = ad::nn::RNNParams::FromSerialized(in);
    seq.fc = ad::nn::FullyConnParams::FromSerialized(in);
    return seq;
}

