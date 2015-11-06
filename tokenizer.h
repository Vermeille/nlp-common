#pragma once

#include <vector>
#include <string>

#include "document.h"

enum class Lang {
    FR
};

struct Tokenizer {
    static std::vector<WordFeatures> FR(const std::string& str);
};
