#include <locale>
#include <iostream>
#include <string>
#include <cctype>

#include "tokenizer.h"
#include "utils.h"

std::vector<WordFeatures> Tokenizer::FR(const std::wstring& str) {
    std::vector<WordFeatures> sentence;
    size_t idx = 0;

    std::locale fr("fr_FR.UTF-8");

    std::wstring tok;
    while (idx < str.size()) {
        if (isspace(str[idx], fr)) {
            // skip & end word
            if (tok != L"") {
                sentence.emplace_back(tok);
                tok.clear();
            }
        } else if (isalnum(str[idx], fr) || str[idx] == '-') {
            tok += str[idx];
        } else if (str[idx] == '\'') {
            tok += '\'';
            sentence.emplace_back(tok);
            tok.clear();
        } else if (ispunct(str[idx], fr)) {
            sentence.emplace_back(tok);
            tok.clear();
            sentence.emplace_back(std::wstring() + str[idx]);
        }

        ++idx;
    }
    if (tok != L"") {
        sentence.emplace_back(tok);
    }
    return sentence;
}

std::vector<WordFeatures> Tokenizer::FR(const std::string& str) {
    return FR(ToWideStr(str));
}

std::vector<WordFeatures> Tokenizer::CharLevel(const std::wstring& str) {
    std::locale fr("fr_FR.UTF-8");
    std::vector<WordFeatures> sentence;
    for (auto c : str) {
        if (isascii(c) || isalnum(c, fr)) {
            sentence.push_back(std::wstring() + c);
        }
    }
    return sentence;
}

std::vector<WordFeatures> Tokenizer::CharLevel(const std::string& str) {
    return Tokenizer::CharLevel(ToWideStr(str));
}
