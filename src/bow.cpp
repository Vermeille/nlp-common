#include <glog/logging.h>

#include "bow.h"

static const unsigned int kNotFound = -1;
static const double kLearningRate = 0.001;

BagOfWords::BagOfWords(size_t in_sz, size_t out_sz)
        : w_weights_(std::make_shared<Eigen::MatrixXd>(out_sz, in_sz)),
        b_weights_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        input_size_(in_sz),
        output_size_(out_sz) {
}

BagOfWords::BagOfWords()
        : w_weights_(std::make_shared<Eigen::MatrixXd>(0, 0)),
        b_weights_(std::make_shared<Eigen::MatrixXd>(0, 1)),
        input_size_(0),
        output_size_(0) {
}

void BagOfWords::Init() {
    w_weights_->setZero();
    b_weights_->setZero();
}

double BagOfWords::ComputeNLL(double* probas) const {
    double nll = 0;
    for (size_t i = 0; i < output_size_; ++i) {
        nll += std::log(probas[i]);
    }
    return -nll;
}

ad::Var BagOfWords::ComputeModel(
        ad::ComputationGraph& g, ad::Var& w, ad::Var& b,
        const std::vector<WordFeatures>& ws) const {
    Eigen::MatrixXd input(input_size_, 1);
    input.setZero();

    for (auto& wf : ws) {
        // one hot encode each word
        if (wf.idx < input_size_) {
            input(wf.idx, 0) = 1;
        }
    }

    ad::Var x = g.CreateParam(input);

    return ad::Softmax(w * x + b);
}

Eigen::MatrixXd BagOfWords::ComputeClass(const std::vector<WordFeatures>& ws) const {
    ad::ComputationGraph g;
    ad::Var w = g.CreateParam(w_weights_);
    ad::Var b = g.CreateParam(b_weights_);

    return ComputeModel(g, w, b, ws).value();
}

int BagOfWords::Train(const Document& doc) {
    double nll = 0;
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        using namespace ad;

        Eigen::MatrixXd y_mat(output_size_, 1);
        y_mat.setZero();
        y_mat(ex.output, 0) = 1;

        ComputationGraph g;
        Var w = g.CreateParam(w_weights_);
        Var b = g.CreateParam(b_weights_);
        Var y = g.CreateParam(y_mat);

        Var h = ComputeModel(g, w, b, ex.inputs);

        Var J = ad::CrossEntropy(y, h);

        ad::opt::SGD sgd(0.01);
        g.BackpropFrom(J);
        g.Update(sgd, {&w, &b});

        Eigen::MatrixXd& h_mat = h.value();
        Eigen::MatrixXd::Index max_row, max_col;
        h_mat.maxCoeff(&max_row, &max_col);
        Label predicted = max_row;
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;

        nll += J.value()(0, 0);
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string BagOfWords::Serialize() const {
#if 0
    std::ostringstream out;
    out << input_size_ << " " << output_size_ << std::endl;

    for (size_t w = 0; w < input_size_; ++w) {
        for (size_t i = 0; i < output_size_; ++i) {
            out << weight(i, w) << " ";
        }
        out << std::endl;
    }
    return out.str();
#endif
    return "";
}

BagOfWords BagOfWords::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz = 0;
    size_t out_sz = 0;
    in >> in_sz >> out_sz;

    BagOfWords bow(in_sz, out_sz);

#if 0
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
#endif

    return bow;
}

void BagOfWords::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    Eigen::MatrixXd& w_mat = *w_weights_;
    w_mat.conservativeResize(output_size_, in);

    for (int row = 0, nb_rows = w_weights_->rows(); row < nb_rows; ++row) {
        for (size_t i = input_size_; i < in; ++i) {
            w_mat(row, i) = 0;
        }
    }
    input_size_ = in;
}

void BagOfWords::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    Eigen::MatrixXd& w_mat = *w_weights_;
    w_mat.conservativeResize(out, input_size_);
    Eigen::MatrixXd& b_mat = *b_weights_;
    b_mat.conservativeResize(out, 1);

    for (unsigned row = output_size_; row < out; ++row) {
        for (unsigned  col = 0, nb_cols = w_weights_->cols(); col < nb_cols; ++col) {
            w_mat(row, col) = 0;
        }
    }

    for (unsigned row = output_size_; row < out; ++row) {
        b_mat(row, 0) = 0;
    }
}

