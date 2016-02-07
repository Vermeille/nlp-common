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

ad::nn::NeuralOutput<ad::Var> BagOfWords::ComputeModel(
        ad::ComputationGraph& g,
        const std::vector<WordFeatures>& ws) const {
    using namespace ad;
    std::vector<int> idxs;
    idxs.reserve(ws.size());
    std::transform(ws.begin(), ws.end(), std::back_inserter(idxs),
            [](const WordFeatures& wf) { return wf.idx; }
        );

    return nn::Map(Softmax, nn::Sum(nn::HashtableQuery(g, words_, idxs)));
}

Eigen::MatrixXd BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    return ComputeModel(g, ws).out.value();
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

        Var J = ad::CrossEntropy(y, h.out);

        opt::SGD sgd(0.1);
        g.BackpropFrom(J);
        g.Update(sgd, *h.params);

        Label predicted = utils::OneHotVectorDecode(h.out.value());
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

