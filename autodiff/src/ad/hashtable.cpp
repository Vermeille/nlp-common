#include "hashtable.h"

#include "operators.h"
#include "helpers.h"

namespace ad {
namespace nn {

Hashtable::Hashtable(size_t wordvec_size, size_t vocab_size, double init)
        : wordvec_size_(wordvec_size),
        vocab_size_(vocab_size) {
    w_.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        w_.push_back(std::make_shared<Param>(wordvec_size, 1, init));
    }
}

std::string Hashtable::Serialize() const {
    std::ostringstream out;
    out << "HASHTABLE\n"
        << wordvec_size_ << " " << vocab_size_ << "\n";
    for (auto& w : w_) {
        ad::utils::WriteMatrixTxt(w->value(), out);;
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

    Hashtable hash(wordvec_size, 0);
    hash.vocab_size_ = vocab_size;
    for (size_t i = 0; i < vocab_size; ++i) {
        hash.w_.push_back(
                std::make_shared<Param>(ad::utils::ReadMatrixTxt(in)));
    }
    return hash;
}

void Hashtable::ResizeVectors(size_t size, double init) {
    for (auto& w : w_) {
        utils::RandomExpandMatrix(w->value(), size, 1, -init, init);
    }
}

void Hashtable::ResizeVocab(size_t size, double init) {
    for (unsigned col = vocab_size_; col < size; ++col) {
        w_.push_back(std::make_shared<Param>(wordvec_size_, 1, init));
    }
    vocab_size_ = size;
}

Var Hashtable::MakeVarFor(ComputationGraph& g, size_t idx, bool learnable) const {
    return g.CreateParam(w_[idx], learnable);
}

std::shared_ptr<Param> Hashtable::Get(size_t idx) const {
    return w_[idx];
}

} // nn
} // ad
