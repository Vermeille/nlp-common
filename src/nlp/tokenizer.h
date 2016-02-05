#pragma once

#include <vector>
#include <string>

#include "document.h"

struct Tokenizer {
    static std::vector<WordFeatures> FR(const std::string& str);
    static std::vector<WordFeatures> FR(const std::wstring& str);
};
