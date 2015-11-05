#pragma once

#include <string>
#include <vector>

typedef unsigned int Label;

struct WordFeatures {
    size_t idx;
    std::string str;

    WordFeatures(const std::string& s) : str(s), idx(0) {}
};

struct FeaturesExtractor {
    static std::vector<WordFeatures> Do(const std::vector<std::string>& sentence);
};


