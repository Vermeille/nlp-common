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

ad::nn::NeuralOutput<ad::Var> SequenceClassifier::ComputeModel(
        ad::ComputationGraph& g,
        const std::vector<WordFeatures>& ws) const {
    std::vector<ad::Var> words;
    std::transform(ws.begin(), ws.end(), std::back_inserter(words),
            [&](const WordFeatures& wf) {
                return words_.MakeVarFor(g, wf.idx);
            }
    );
    auto a1 = ad::nn::InputLayer(words);
    auto a2 = encoder_.Compute(a1);
    auto a3 = decoder_.Compute(a2);
    a3.out = ad::Softmax(a3.out);
    return a3;
}

Eigen::MatrixXd SequenceClassifier::ComputeClass(
        const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    return ComputeModel(g, ws).out.value();
}

double SequenceClassifier::Train(const Document& doc) {
    int nb_corrects = 0;

    for (auto& ex : doc.examples) {
        using namespace ad;
        ComputationGraph g;

        Eigen::MatrixXd y_mat
            = utils::OneHotColumnVector(ex.output, output_size_);

        Var y = g.CreateParam(y_mat);
        auto h = ComputeModel(g, ex.inputs);
        Var J = ad::MSE(y, h.out) + 1e-4 * nn::L2ForAllParams(h);

        opt::SGD sgd(0.1);
        g.BackpropFrom(J);
        g.Update(sgd, *h.params);

        Label predicted = utils::OneHotVectorDecode(h.out.value());
        nb_corrects += predicted == ex.output ? 1 : 0;
        std::cout << "nll = " << J.value()(0, 0) << "\n";
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

