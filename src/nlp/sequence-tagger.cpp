#include <glog/logging.h>

#include "sequence-tagger.h"

static const unsigned int kNotFound = -1;
static constexpr double kLearningRate = 0.01;

static double randr(float from, float to) {
    double distance = to - from;
    return ((double)rand() / ((double)RAND_MAX + 1) * distance) + from;
}

SequenceTagger::SequenceTagger(
            size_t in_sz, size_t out_sz,
            size_t start_word, size_t start_label,
            size_t stop_word, size_t stop_label)
        : woo_(std::make_shared<Eigen::MatrixXd>(out_sz, out_sz)),
        b_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        input_size_(in_sz),
        output_size_(out_sz),
        start_word_(start_word),
        start_label_(start_label),
        stop_word_(stop_word),
        stop_label_(stop_label) {

    wox_.reserve(in_sz);
    for (int i = 0; i < in_sz; i++) {
        wox_.push_back(std::make_shared<Eigen::MatrixXd>(out_sz, 1));
        Eigen::MatrixXd& wox = *wox_.back();
        for (size_t j = 0; j < out_sz; ++j) {
            wox(j, 0) = randr(-1, 1);
        }
    }

    Eigen::MatrixXd& woo = *woo_;
    for (size_t i = 0; i < out_sz; ++i) {
        for (size_t j = 0; j < out_sz; ++j) {
            woo(i, j) = randr(-1, 1);
        }
    }
}

SequenceTagger::SequenceTagger()
        : woo_(std::make_shared<Eigen::MatrixXd>(0, 0)),
        b_(std::make_shared<Eigen::MatrixXd>(0, 1)),
        input_size_(0),
        output_size_(0) {
}

void SequenceTagger::Compute(std::vector<WordFeatures>& ws) {
    std::vector<ad::Var> woxes;

    ad::ComputationGraph g;
    ad::Var woo = g.CreateParam(woo_);
    ad::Var b = g.CreateParam(b_);


    const auto& outs = ComputeModel(g, woxes, woo, b, ws.begin(), ws.end());
    for (size_t i = 0; i < outs.size(); ++i) {
        Eigen::MatrixXd::Index label, dummy;
        outs[i].value().maxCoeff(&label, &dummy);

        ws[i].pos = label;
    }
}

std::vector<ad::Var> SequenceTagger::ComputeModel(ad::ComputationGraph& g,
        std::vector<ad::Var>& woxes, ad::Var woo, ad::Var b,
        std::vector<WordFeatures>::const_iterator begin,
        std::vector<WordFeatures>::const_iterator end) const {
    using namespace ad;

    std::vector<Var> outputs;

    Eigen::MatrixXd zero(output_size_, 1);
    zero.setZero();

    Var prev = g.CreateParam(zero);

    while (begin != end) {
        Var wox = g.CreateParam(wox_[begin->idx]);

        Var z = wox + woo * prev + b;
        prev = Sigmoid(z);
        outputs.push_back(Softmax(z));
        woxes.push_back(wox);

        ++begin;
    }
    return outputs;
}

int SequenceTagger::Train(const Document& doc) {
    using namespace ad;
    double nll = 0;
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        auto begin = ex.inputs.begin();
        auto end = begin + 1;
        for (int i = 0; i < ex.inputs.size(); ++i) {
            auto& wf = ex.inputs[i];
            std::vector<Var> woxes;

            ComputationGraph g;
            Var woo = g.CreateParam(woo_);
            Var b = g.CreateParam(b_);

            Eigen::MatrixXd yt_mat(output_size_, 1);
            yt_mat.setZero();
            if (wf.pos < output_size_ && wf.pos != -1) {
                yt_mat(wf.pos, 0) = 1;
            }
            Var yt = g.CreateParam(yt_mat);

            auto outs = ComputeModel(g, woxes, woo, b, begin, end);

            Var Jt = MSE(outs.back(), yt);

            nll += Jt.value().sum();

            g.BackpropFrom(Jt);

            opt::SGD sgd(0.1);
            g.Update(sgd, {&woo, &b});
            for (auto w : woxes) {
                g.Update(sgd, {&w});
            }
            ++end;

            Eigen::MatrixXd::Index predicted, dummy;
            outs.back().value().maxCoeff(&predicted, &dummy);

            nb_correct += predicted == ex.inputs[i].pos ? 1 : 0;
            ++nb_tokens;
        }
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string SequenceTagger::Serialize() const {
#if 0
    std::ostringstream out;
    out << input_size_ << " " << output_size_ << " " <<
        start_word_ << " " << start_label_ << " " <<
        stop_word_ << " " << stop_label_ << std::endl;

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

SequenceTagger SequenceTagger::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz = 0;
    size_t out_sz = 0;
    size_t start_word, start_label, stop_word, stop_label;
    in >> in_sz >> out_sz >>
        start_word >> start_label >>
        stop_word >> stop_label;

    SequenceTagger bow(in_sz, out_sz,
            start_word, start_label,
            stop_word, stop_label);

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

void SequenceTagger::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    // fill in new rightmost columns
    for (unsigned col = input_size_; col < in; ++col) {
        wox_.push_back(std::make_shared<Eigen::MatrixXd>(output_size_, 1));
        Eigen::MatrixXd& wox = *wox_.back();
        for (size_t j = 0; j < output_size_; ++j) {
            wox(j, 0) = randr(-1, 1);
        }
    }
    input_size_ = in;
}

void SequenceTagger::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    // fill-in new bottom rows
    for (auto& w : wox_) {
        Eigen::MatrixXd& wox = *w;
        wox.conservativeResize(out, 1);
        for (unsigned row = output_size_; row < out; ++row) {
            wox(row, 0) = randr(-1, 1);
        }
    }

    Eigen::MatrixXd& woo = *woo_;;
    woo.conservativeResize(out, out);
    // new rows
    for (unsigned row = output_size_; row < out; ++row) {
        for (unsigned col = 0; col < output_size_; ++col) {
            woo(row, col) = randr(-1, 1);
        }
    }

    // new cols
    for (unsigned row = 0; row < out; ++row) {
        for (unsigned col = output_size_; col < out; ++col) {
            woo(row, col) = randr(-1, 1);
        }
    }

    Eigen::MatrixXd& b = *b_;
    b.conservativeResize(out, 1);
    for (unsigned row = output_size_; row < out; ++row) {
        b(row, 0) = randr(-1, 1);
    }


    output_size_ = out;
}

