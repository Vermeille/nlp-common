#pragma once

#include <vector>
#include <string>
#include <boost/bimap.hpp>

typedef unsigned int Label;

class LabelSet {
  public:
    Label GetLabel(const std::string& str) {
        return labels_.insert(decltype (labels_)::value_type(str, labels_.size())).first->right;
    }

    void AddLabel(const std::string& str) { GetLabel(str); }

    std::string GetString(Label pos) const {
        return labels_.right.at(pos);
    }

    size_t size() const { return labels_.size(); }

  private:
    boost::bimap<std::string, Label> labels_;
};

typedef unsigned int Label;

struct WordFeatures {
    std::string str;
    size_t idx;

    WordFeatures(const std::string& s) : str(s), idx(0) {}
};

struct TrainingExample {
    std::vector<WordFeatures> inputs;
    Label output;
};

struct Document {
    std::vector<TrainingExample> examples;
};

