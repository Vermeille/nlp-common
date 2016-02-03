#include <glog/logging.h>

#include "sequence-tagger.h"

SequenceTagger::SequenceTagger(size_t in_sz, size_t out_sz)
        : woo_(std::make_shared<Eigen::MatrixXd>(out_sz, out_sz)),
        b_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        input_size_(in_sz),
        output_size_(out_sz) {

    wox_.reserve(in_sz);
    for (int i = 0; i < in_sz; i++) {
        wox_.push_back(std::make_shared<Eigen::MatrixXd>(out_sz, 1));
        ad::utils::RandomInit(*wox_.back(), -1, 1);
    }

    ad::utils::RandomInit(*woo_, -1, 1);
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
        ws[i].pos = ad::utils::OneHotVectorDecode(outs[i].value());
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


            Eigen::MatrixXd yt_mat
                = ad::utils::OneHotColumnVector(wf.pos, output_size_);
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

            Label predicted = ad::utils::OneHotVectorDecode(outs.back().value());

            nb_correct += predicted == ex.inputs[i].pos ? 1 : 0;
            ++nb_tokens;
        }
    }
    return  nb_correct * 100 / nb_tokens;
}

std::string SequenceTagger::Serialize() const {
    std::ostringstream out;
    out << input_size_ << " " << output_size_ << " " << std::endl;

    for (auto& w : wox_) {
        ad::utils::WriteMatrix(*w, out);;
    }

    ad::utils::WriteMatrix(*woo_, out);
    ad::utils::WriteMatrix(*b_, out);
    return out.str();
}

SequenceTagger SequenceTagger::FromSerialized(std::istream& in) {
    std::string tok;
    size_t in_sz = 0;
    size_t out_sz = 0;
    in >> in_sz >> out_sz;

    SequenceTagger bow(in_sz, out_sz);

    for (int i = 0; i < in_sz; ++i) {
        bow.wox_.push_back(std::make_shared<Eigen::MatrixXd>(
                    ad::utils::ReadMatrix(in)));
    }

    bow.woo_ = std::make_shared<Eigen::MatrixXd>(ad::utils::ReadMatrix(in));
    bow.b_ = std::make_shared<Eigen::MatrixXd>(ad::utils::ReadMatrix(in));
    return bow;
}

void SequenceTagger::ResizeInput(size_t in) {
    if (in <= input_size_) {
        return;
    }

    for (unsigned col = input_size_; col < in; ++col) {
        wox_.push_back(std::make_shared<Eigen::MatrixXd>(output_size_, 1));
        ad::utils::RandomInit(*wox_.back(), -1, 1);
    }
    input_size_ = in;
}

void SequenceTagger::ResizeOutput(size_t out) {
    if (out <= output_size_) {
        return;
    }

    for (auto& w : wox_) {
        ad::utils::RandomExpandMatrix(*w, out, 1, -1, 1);
    }

    ad::utils::RandomExpandMatrix(*woo_, out, out, -1, 1);
    ad::utils::RandomExpandMatrix(*b_, out, 1, -1, 1);

    output_size_ = out;
}

