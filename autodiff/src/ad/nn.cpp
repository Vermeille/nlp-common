#include "nn.h"

#include "operators.h"

namespace ad {
namespace nn {

Var L2(const Var& x) {
    return Mean(EltSquare(x));
}

Hashtable::Hashtable(size_t vocab_size, size_t wordvec_size)
        : w_(vocab_size),
        wordvec_size_(wordvec_size),
        vocab_size_(vocab_size) {
    for (size_t i = 0; i < vocab_size; i++) {
        w_.push_back(std::make_shared<Eigen::MatrixXd>(wordvec_size, 1));
        ad::utils::RandomInit(*w_.back(), -1, 1);
    }
}

std::string Hashtable::Serialize() const {
    std::ostringstream out;
    out << "HASHTABLE\n"
        << wordvec_size_ << " " << vocab_size_ << "\n";
    for (auto& w : w_) {
        ad::utils::WriteMatrixTxt(*w, out);;
    }

    return out.str();
}

Hashtable Hashtable::FromSerialized(std::istream& in) {
    std::string magic;
    size_t vocab_size;
    size_t wordvec_size;
    in >> magic
        >> wordvec_size >> vocab_size;

    if (magic != "HASHTABLE") {
        throw std::runtime_error("Magic HASHTABLE not found, '" + magic
                + "' found instead");
    }

    Hashtable hash(0, wordvec_size);
    hash.vocab_size_ = vocab_size;
    for (size_t i = 0; i < vocab_size; ++i) {
        hash.w_.push_back(
            std::make_shared<Eigen::MatrixXd>(ad::utils::ReadMatrixTxt(in)));
    }
    return hash;
}

void Hashtable::ResizeVectors(size_t size) {
    for (auto& w : w_) {
        utils::RandomExpandMatrix(*w, size, 1, -1, 1);
    }
}

void Hashtable::ResizeVocab(size_t size) {
    for (unsigned col = vocab_size_; col < size; ++col) {
        w_.push_back(std::make_shared<Eigen::MatrixXd>(wordvec_size_, 1));
        ad::utils::RandomInit(*w_.back(), -1, 1);
    }
    vocab_size_ = size;
}

Var Hashtable::MakeVarFor(ComputationGraph& g, size_t idx) const {
    return g.CreateParam(w_[idx]);
}

std::shared_ptr<Eigen::MatrixXd> Hashtable::Get(size_t idx) const {
    return w_[idx];
}

} // nn
} // ad
