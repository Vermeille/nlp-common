#pragma once

#include <vector>
#include <string>

#include "document.h"

struct Tokenizer {
    static std::vector<WordFeatures> FR(const std::string& str);
    static std::vector<WordFeatures> FR(const std::wstring& str);

    static std::vector<WordFeatures> CharLevel(const std::string& str);
    static std::vector<WordFeatures> CharLevel(const std::wstring& str);
};
