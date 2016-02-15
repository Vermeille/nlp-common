#include "mrnn.h"

#include <stdexcept>

#include "../helpers.h"
#include "../operators.h"

namespace ad {
namespace nn {

DiscreteMRNNParams::DiscreteMRNNParams(int out_sz, size_t in_sz, double init) :
        bh_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)),
        h_(std::make_shared<Eigen::MatrixXd>(out_sz, 1)) {
    for (size_t i = 0; i < in_sz; ++i) {
        whx_.push_back(std::make_shared<Eigen::MatrixXd>(out_sz, 1));
        ad::utils::RandomInit(*whx_.back() , -init, init);
    }

    for (size_t i = 0; i < in_sz; ++i) {
        whh_.push_back(std::make_shared<Eigen::MatrixXd>(out_sz, out_sz));
        ad::utils::RandomInit(*whh_.back() , -init, init);
    }
    ad::utils::RandomInit(*bh_ , -init, init);
    h_->setZero();
}

void DiscreteMRNNParams::ResizeInput(size_t size, double init) {
    size_t out_sz = bh_->rows();
    while (whx_.size() < size) {
        whx_.push_back(std::make_shared<Eigen::MatrixXd>(out_sz, 1));
        ad::utils::RandomInit(*whx_.back() , -init, init);
    }
    while (whh_.size() < size) {
        whh_.push_back(std::make_shared<Eigen::MatrixXd>(out_sz, out_sz));
        ad::utils::RandomInit(*whh_.back() , -init, init);
    }
}

void DiscreteMRNNParams::ResizeOutput(size_t size, double init) {
    for (auto& v : whx_) {
        utils::RandomExpandMatrix(*v, size, 1, -init, init);
    }
    for (auto& v : whh_) {
        utils::RandomExpandMatrix(*v, size, size, -init, init);
    }
    utils::RandomExpandMatrix(*h_, size, 1, -init, init);
    utils::RandomExpandMatrix(*bh_, size, 1, -init, init);
}

void DiscreteMRNNParams::Serialize(std::ostream& out) const {
    out << "MRNN\n";
    out << whx_.size() << "\n";
    for (auto& v : whx_) {
        utils::WriteMatrixTxt(*v, out);
    }
    for (auto& v : whh_) {
        utils::WriteMatrixTxt(*v, out);
    }
    utils::WriteMatrixTxt(*bh_, out);
    utils::WriteMatrixTxt(*h_, out);
}

DiscreteMRNNParams DiscreteMRNNParams::FromSerialized(std::istream& in) {
    std::string magic;
    in >> magic;
    if (magic != "MRNN") {
        throw std::runtime_error("Not a MRNN layer, but " + magic);
    }

    size_t in_sz;
    in >> in_sz;
    DiscreteMRNNParams rnn(0, 0);
    for (size_t i = 0; i < in_sz; ++i) {
        rnn.whx_.push_back(
                std::make_shared<Eigen::MatrixXd>(utils::ReadMatrixTxt(in)));
    }
    for (size_t i = 0; i < in_sz; ++i) {
        rnn.whh_.push_back(
                std::make_shared<Eigen::MatrixXd>(utils::ReadMatrixTxt(in)));
    }
    *rnn.h_ = utils::ReadMatrixTxt(in);
    *rnn.bh_ = utils::ReadMatrixTxt(in);
    return rnn;
}

DiscreteMRNNLayer::DiscreteMRNNLayer(
            ComputationGraph& g,
            const DiscreteMRNNParams& params,
            bool learnable) :
        bh_(g.CreateParam(params.bh_, learnable)),
        h_(g.CreateParam(params.h_, learnable)) {
    for (size_t i = 0, end = params.whx_.size(); i < end; ++i) {
        whx_.emplace_back(g.CreateParam(params.whx_[i], learnable));
        whh_.emplace_back(g.CreateParam(params.whh_[i], learnable));
    }
    used_vars_.push_back(bh_);
    used_vars_.push_back(h_);
}

Var DiscreteMRNNLayer::Step(int x) {
    used_vars_.push_back(whx_[x]);
    used_vars_.push_back(whh_[x]);
    return h_ = Sigmoid(whx_[x] + whh_[x] * h_ + bh_);
}

std::vector<Var> DiscreteMRNNLayer::Params() const {
    return used_vars_;
}

} // nn
} // ad