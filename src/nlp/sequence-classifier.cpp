#include <iterator>

#include "sequence-classifier.h"
#include <ad/ad.h>

SequenceClassifier::SequenceClassifier(
        size_t out_sz, size_t hidden_size, size_t wordvec_size, size_t vocab_size)
    : words_(wordvec_size, vocab_size),
    encoder_(hidden_size, wordvec_size),
    decoder_(out_sz, hidden_size),
    output_size_(out_sz) {
}

ad::Var SequenceClassifier::ComputeModel(
        ad::ComputationGraph& g,
        const std::vector<WordFeatures>& ws) const {
    using namespace ad;
    nn::RNNLayer encoder(g, encoder_);
    nn::FullyConnLayer decoder(g, decoder_);

    Var encoded(nullptr);
    for (auto& w : ws) {
        encoded = encoder.Step(words_.MakeVarFor(g, w.idx));
    }
    return ad::Softmax(decoder.Compute(encoded));
}

Eigen::MatrixXd SequenceClassifier::ComputeClass(
        const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    return ComputeModel(g, ws).value();
}

double SequenceClassifier::Train(const Document& doc) {
    int nb_corrects = 0;

    for (auto& ex : doc.examples) {
        if (ex.inputs.empty()) {
            continue;
        }

        using namespace ad;
        ComputationGraph g;

        Eigen::MatrixXd y_mat
            = utils::OneHotColumnVector(ex.output, output_size_);

        Var y = g.CreateParam(y_mat);
        auto h = ComputeModel(g, ex.inputs);
        Var J = ad::MSE(y, h);

        opt::SGD sgd(0.1);
        g.BackpropFrom(J, 5);
        g.Update(sgd);

        Label predicted = utils::OneHotVectorDecode(h.value());
        nb_corrects += predicted == ex.output ? 1 : 0;
    }

    return 100.0 * nb_corrects / doc.examples.size();
}

void SequenceClassifier::ResizeInput(size_t in) {
    words_.ResizeVocab(in);
}

void SequenceClassifier::ResizeOutput(size_t out) {
    decoder_.ResizeOutput(out);
    output_size_ = out;
}

std::string SequenceClassifier::Serialize() const {
    std::ostringstream oss;
    oss << "SEQUENCE-CLASSIFIER\n";
    oss << words_.Serialize();
    encoder_.Serialize(oss);
    decoder_.Serialize(oss);
    return oss.str();
}

SequenceClassifier SequenceClassifier::FromSerialized(std::istream& file) {
    std::string magic;
    file >> magic;
    if (magic != "SEQUENCE-CLASSIFIER") {
        throw std::runtime_error("Magic is not SEQUENCE-CLASSIFIER");
    }
    SequenceClassifier seq(0, 0, 0, 0);
    seq.words_ = decltype (seq.words_)::FromSerialized(file);
    seq.encoder_ = decltype (seq.encoder_)::FromSerialized(file);
    seq.decoder_ = decltype (seq.decoder_)::FromSerialized(file);
    return seq;
}

