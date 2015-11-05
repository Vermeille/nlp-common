#include "tokenizer.h"

template <>
std::vector<std::string> Tokenizer::Do<Lang::FR>(const std::string& str) {
    std::vector<std::string> sentence;
    int idx = 0;

    std::string tok;
    while (idx < str.size()) {
        if (isspace(str[idx])) {
            // skip & end word
            if (tok != "") {
                sentence.push_back(tok);
                tok.clear();
            }
        } else if (isalpha(str[idx]) || str[idx] == '-') {
            tok += str[idx];
        } else if (str[idx] == '\'') {
            tok += '\'';
            sentence.push_back(tok);
            tok.clear();
        } else if (ispunct(str[idx])) {
            sentence.push_back(std::string() + str[idx]);
        }

        ++idx;
    }
    return sentence;
}
