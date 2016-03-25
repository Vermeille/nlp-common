#include <iterator>

#include "sequence-classifier.h"
#include <ad/ad.h>

SequenceClassifier::SequenceClassifier(
        size_t out_sz, size_t hidden_size, size_t wordvec_size, size_t vocab_size)
    : words(wordvec_size, vocab_size),
    encoder(hidden_size, wordvec_size),
    decoder(out_sz, hidden_size),
    output_size_(out_sz) {
}

ad::Var SequenceClassifierGraph::Step(
        ad::ComputationGraph& g,
        const std::vector<WordFeatures>& ws) {
    using namespace ad;
    Var encoded(nullptr);
    for (auto& w : ws) {
        encoded = encoder_.Step(words_.MakeVarFor(g, w.idx));
    }
    return ad::Softmax(decoder_.Compute(encoded));
}

ad::Matrix& SequenceClassifier::ComputeClass(
        const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    SequenceClassifierGraph graph(g, *this, input_size_, output_size_);
    return graph.Step(g, ws).value();
}

double SequenceClassifier::Train(const Document& doc) {
    for (auto& ex : doc.examples) {
        if (ex.inputs.empty()) {
            continue;
        }

        using namespace ad;
        ad::train::FeedForwardTrainer trainer(new ad::opt::Adagrad(0.1));

        trainer.Step(ex.inputs, ex.output,
                [&](ad::ComputationGraph& g) {
                    return SequenceClassifierGraph(g, *this, input_size_, output_size_);
                });
    }

    return Test(doc);
}

double SequenceClassifier::Test(const Document& doc) {
    int nb_corrects = 0;

    for (auto& ex : doc.examples) {
        if (ex.inputs.empty()) {
            continue;
        }

        using namespace ad;
        ad::ComputationGraph g;
        SequenceClassifierGraph seq(g, *this, input_size_, output_size_);

        Var h = seq.Step(g, ex.inputs);
        Label predicted = utils::OneHotVectorDecode(h.value());
        nb_corrects += predicted == ex.output ? 1 : 0;
    }

    return 100.0 * nb_corrects / doc.examples.size();
}
void SequenceClassifier::ResizeInput(size_t in) {
    words.ResizeVocab(in);
}

std::string SequenceClassifier::Serialize() const {
    std::ostringstream oss;
    oss << "SEQUENCE-CLASSIFIER\n";
    oss << words.Serialize();
    encoder.Serialize(oss);
    decoder.Serialize(oss);
    return oss.str();
}

SequenceClassifier SequenceClassifier::FromSerialized(std::istream& file) {
    std::string magic;
    file >> magic;
    if (magic != "SEQUENCE-CLASSIFIER") {
        throw std::runtime_error("Magic is not SEQUENCE-CLASSIFIER");
    }
    SequenceClassifier seq(0, 0, 0, 0);
    seq.words = decltype (seq.words)::FromSerialized(file);
    seq.encoder = decltype (seq.encoder)::FromSerialized(file);
    seq.decoder = decltype (seq.decoder)::FromSerialized(file);
    return seq;
}

