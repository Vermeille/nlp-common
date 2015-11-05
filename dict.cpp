#include "dict.h"

Dictionnary::Dictionnary()
        : max_freq_(0) {
    unk_id_ = GetWordId("_UNK_");
}

size_t Dictionnary::GetWordId(const std::string& w) {
    size_t id = dict_.insert(decltype (dict_)::value_type(w, dict_.size())).first->right;
    if (stats_.size() < id) {
        stats_.resize(id);
        stats_[id] = 1;
    }
    return id;
}

size_t Dictionnary::GetWordIdOrUnk(const std::string& w) {
    auto res = dict_.left.find(w);
    if (res == dict_.left.end()) {
        return unk_id_;
    } else {
        ++stats_[res->second];
        max_freq_ = std::max(max_freq_, stats_[res->second]);
        return res->second;
    }
}

void NGramMaker::Annotate(std::vector<WordFeatures>& sentence) {
    for (auto& s : sentence) {
        s.idx = dict_.GetWordIdOrUnk(s.str);
    }
}

void NGramMaker::Learn(std::vector<WordFeatures>& sentence) {
    for (auto& s : sentence) {
        s.idx = dict_.GetWordId(s.str);
    }
}

