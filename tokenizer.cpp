#include "tokenizer.h"

std::vector<WordFeatures> Tokenizer::FR(const std::string& str) {
    std::vector<WordFeatures> sentence;
    int idx = 0;

    std::string tok;
    while (idx < str.size()) {
        if (isspace(str[idx])) {
            // skip & end word
            if (tok != "") {
                sentence.emplace_back(tok);
                tok.clear();
            }
        } else if (isalpha(str[idx]) || str[idx] == '-') {
            tok += str[idx];
        } else if (str[idx] == '\'') {
            tok += '\'';
            sentence.emplace_back(tok);
            tok.clear();
        } else if (ispunct(str[idx])) {
            sentence.emplace_back(std::string() + str[idx]);
        }

        ++idx;
    }
    return sentence;
}
