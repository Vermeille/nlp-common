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

ad::Var BagOfWords::Step(
        ad::ComputationGraph& g,
        const std::vector<WordFeatures>& ws) const {
    using namespace ad;

    Var sum = words_.MakeVarFor(g, ws[0].idx);
    for (size_t i = 1; i < ws.size(); ++i) {
        sum = sum + words_.MakeVarFor(g, ws[i].idx);
    }

    return sum;
}

Eigen::MatrixXd BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    return ad::Softmax(Step(g, ws)).value();
}

ad::Var BagOfWords::Cost(ad::ComputationGraph& g, ad::Var h, int output_class) {
    using namespace ad;
    Var y = g.CreateParam(utils::OneHotColumnVector(output_class, output_size_));
    return ad::Mean(ad::SoftmaxLoss(h, y))
        + 1e-4 * ad::nn::L2(g.GetAllLearnableVars());
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
    return words_.Get(word)->value()(label, 0);
}

