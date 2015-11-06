#include <glog/logging.h>

#include "bow.h"

static const unsigned int kNotFound = -1;
static const double kLearningRate = 0.001;

BagOfWords::BagOfWords(size_t in_sz, size_t out_sz)
        : input_size_(in_sz),
        output_size_(out_sz) {
}

void BagOfWords::Init() {
    word_weight_.resize(output_size_);
    for (size_t i = 0; i < output_size_; ++i) {
        word_weight_[i].resize(input_size_);
        for (size_t j = 0; j < input_size_; ++j) {
            word_weight_[i][j] = 0;
        }
    }
}

double BagOfWords::WordF(Label target, const WordFeatures& w) const {
    if (w.idx == kNotFound)
        return 0;

    return word_weight_[target][w.idx];
}

void BagOfWords::WordF_Backprop(const WordFeatures& w, Label truth, const double* probabilities) {
    if (w.idx == kNotFound)
        return;

    for (size_t k = 0; k < output_size_; ++k) {
        double target = (truth == k) ? 1 : 0;
        word_weight_[k][w.idx] += kLearningRate * (target - probabilities[k]);
    }
}

double BagOfWords::RunAllFeatures(Label k, const std::vector<WordFeatures>& ws) const {
    double sum = 0;
    for (size_t i = 0; i < ws.size(); ++i) {
        sum += WordF(k, ws[i]);
    }
    return sum;
}

double BagOfWords::ComputeNLL(double* probas) const {
    double nll = 0;
    for (size_t i = 0; i < output_size_; ++i) {
        nll += std::log(probas[i]);
    }
    return -nll;
}

Label BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws, double* probabilities) const {
    double total = 0;
    for (size_t k = 0; k < output_size_; ++k) {
        probabilities[k] = std::exp(RunAllFeatures(k, ws));
        total += probabilities[k];
    }

    int max = 0;
    for (size_t k = 0; k < output_size_; ++k) {
        probabilities[k] /= total;
        if (probabilities[k] > probabilities[max]) {
            max = k;
        }
    }
    return max;
}

void BagOfWords::Backprop(const std::vector<WordFeatures>& ws, Label truth, const double* probabilities) {
    for (size_t i = 0; i < ws.size(); ++i) {
        WordF_Backprop(ws[i], truth, probabilities);
    }
}

int BagOfWords::Train(const Document& doc) {
    double nll = 0;
    std::vector<double> probas(output_size_);
    int nb_correct = 0;
    int nb_tokens = 0;

    word_weight_.resize(output_size_);
    for (size_t i = 0; i < output_size_; ++i) {
        word_weight_[i].resize(input_size_);
    }

    for (auto& ex : doc.examples) {
        Label predicted = ComputeClass(ex.inputs, probas.data());
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;

        nll += ComputeNLL(probas.data());

        Backprop(ex.inputs, ex.output, probas.data());
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string BagOfWords::Serialize() const {
    std::ostringstream out;
    out << input_size_ << " " << output_size_ << std::endl;

    for (size_t w = 0; w < input_size_; ++w) {
        for (size_t i = 0; i < output_size_; ++i) {
            out << weight(i, w) << " ";
        }
        out << std::endl;
    }
    return out.str();
}

BagOfWords BagOfWords::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz, out_sz;
    in >> in_sz >> out_sz;

    BagOfWords bow(in_sz, out_sz);

    for (size_t i = 0; i < bow.output_size_; ++i) {
        bow.word_weight_.emplace_back();
    }

    for (size_t w = 0; w < bow.input_size_; ++w) {
        for (size_t i = 0; i < bow.output_size_; ++i) {
            double score;
            in >> score;
            bow.word_weight_[i].push_back(score);
        }
    }

    return bow;
}

void BagOfWords::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    for (auto& weights : word_weight_) {
        weights.resize(in);
    }
    input_size_ = in;
}

void BagOfWords::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    word_weight_.resize(out);
    for (size_t i = output_size_; i < out; ++i) {
        word_weight_[i].resize(input_size_);
    }
    output_size_ = out;
}

