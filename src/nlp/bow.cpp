#include <glog/logging.h>

#include "bow.h"

static const unsigned int kNotFound = -1;
static const double kLearningRate = 0.001;

BagOfWords::BagOfWords(size_t in_sz, size_t out_sz)
        : w_weights_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
        b_weights_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        input_size_(in_sz),
        output_size_(out_sz) {
    ad::utils::RandomInit(*b_weights_, -1, 1);
    ad::utils::RandomInit(*w_weights_, -1, 1);
}

BagOfWords::BagOfWords()
        : w_weights_(std::make_shared<Eigen::MatrixXd>(0, 0)),
        b_weights_(std::make_shared<Eigen::MatrixXd>(0, 1)),
        input_size_(0),
        output_size_(0) {
}

ad::Var BagOfWords::ComputeModel(
        ad::Var& w, ad::Var& b,
        const std::vector<WordFeatures>& ws) const {
    // sum of words + b
    ad::Var sum = b;
    for (auto& wf : ws) {
        sum = sum + ad::NthCol(w, wf.idx);
    }
    return ad::Softmax(sum);
}

Eigen::MatrixXd BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    ad::Var w = g.CreateParam(w_weights_);
    ad::Var b = g.CreateParam(b_weights_);

    return ComputeModel(w, b, ws).value();
}

int BagOfWords::Train(const Document& doc) {
    double nll = 0;
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        using namespace ad;

        Eigen::MatrixXd y_mat
            = utils::OneHotColumnVector(ex.output, output_size_);

        ComputationGraph g;
        Var w = g.CreateParam(w_weights_);
        Var b = g.CreateParam(b_weights_);
        Var y = g.CreateParam(y_mat);

        Var h = ComputeModel(w, b, ex.inputs);

        Var J = ad::CrossEntropy(y, h);

        opt::SGD sgd(0.1);
        g.BackpropFrom(J);
        g.Update(sgd, {&w, &b});

        Label predicted = utils::OneHotVectorDecode(h.value());
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;

        nll += J.value()(0, 0);
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string BagOfWords::Serialize() const {
    std::ostringstream out;

    out << input_size_ << " " << output_size_ << std::endl;

    auto& b_mat = *b_weights_;
    auto& w_mat = *w_weights_;

    for (size_t w = 0; w < output_size_; ++w) {
        for (size_t i = 0; i < input_size_; ++i) {
            out << w_mat(w, i) << " ";
        }
        out << std::endl;
    }

    for (size_t w = 0; w < output_size_; ++w) {
        out << b_mat(w, 0) << "\n";
    }

    return out.str();
}

BagOfWords BagOfWords::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz = 0;
    size_t out_sz = 0;
    in >> in_sz >> out_sz;

    BagOfWords bow(in_sz, out_sz);
    auto& b_mat = *bow.b_weights_;
    auto& w_mat = *bow.w_weights_;

    for (size_t w = 0; w < bow.output_size_; ++w) {
        for (size_t i = 0; i < bow.input_size_; ++i) {
            double score;
            in >> score;
            w_mat(w, i) = score;
        }
    }

    for (size_t w = 0; w < bow.output_size_; ++w) {
        double score;
        in >> score;
        b_mat(w, 0) = score;
    }

    return bow;
}

void BagOfWords::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    ad::utils::RandomExpandMatrix(*w_weights_, output_size_, in, -1, 1);

    input_size_ = in;
}

void BagOfWords::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    ad::utils::RandomExpandMatrix(*w_weights_, out, input_size_, -1, 1);
    ad::utils::RandomExpandMatrix(*b_weights_, out, 1, -1, 1);

    output_size_ = out;
}

double BagOfWords::weights(size_t label, size_t word) const {
    return (*w_weights_)(label, word);
}

Eigen::MatrixXd& BagOfWords::weights() const {
    return *w_weights_;
}

double BagOfWords::apriori(size_t label) const {
    return (*b_weights_)(label, 0);
}
