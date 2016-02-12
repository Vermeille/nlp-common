#include <glog/logging.h>

#include "bow.h"

static const unsigned int kNotFound = -1;
static const double kLearningRate = 0.001;

BagOfWords::BagOfWords(size_t in_sz, size_t out_sz)
        : words_(out_sz, in_sz),
        output_size_(out_sz) {
}

BagOfWords::BagOfWords()
        : words_(0, 0),
        output_size_(0) {
}

ad::Var BagOfWords::ComputeModel(
        ad::ComputationGraph& g,
        const std::vector<WordFeatures>& ws) const {
    using namespace ad;

    Var sum = words_.MakeVarFor(g, ws[0].idx);
    for (size_t i = 1; i < ws.size(); ++i) {
        sum = sum + words_.MakeVarFor(g, ws[i].idx);
    }

    return Softmax(sum);
}

Eigen::MatrixXd BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    return ComputeModel(g, ws).value();
}

int BagOfWords::Train(const Document& doc) {
    double nll = 0;
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        using namespace ad;

        ComputationGraph g;
        Var y = g.CreateParam(utils::OneHotColumnVector(ex.output, output_size_));

        auto h = ComputeModel(g, ex.inputs);

        Var J = ad::CrossEntropy(y, h);

        opt::SGD sgd(0.1);
        g.BackpropFrom(J, 5);
        g.Update(sgd);

        Label predicted = utils::OneHotVectorDecode(h.value());
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;

        nll += J.value()(0, 0);
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string BagOfWords::Serialize() const {
    return words_.Serialize();
}

BagOfWords BagOfWords::FromSerialized(std::istream& in) {
    BagOfWords bow;
    bow.words_ = ad::nn::Hashtable::FromSerialized(in);
    return bow;
}

void BagOfWords::ResizeInput(size_t in) {
    words_.ResizeVocab(in);
}

void BagOfWords::ResizeOutput(size_t out) {
    words_.ResizeVectors(out);
    output_size_ = out;
}

double BagOfWords::weights(size_t label, size_t word) const {
    return (*words_.Get(word))(label, 0);
}

