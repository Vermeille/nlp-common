#pragma once

#include <vector>
#include <string>

enum class Lang {
    FR
};

struct Tokenizer {
    template <enum Lang>
    static std::vector<std::string> Do(const std::string& str);
};
