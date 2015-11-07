#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <boost/bimap.hpp>

#include "document.h"

class BagOfWords {
    std::vector<std::vector<double>> word_weight_;
    size_t input_size_;
    size_t output_size_;

    void Init();

    double WordF(Label target, const WordFeatures& w) const;
    void WordF_Backprop(const WordFeatures& w, Label truth, const double* probabilities);
    double RunAllFeatures(Label k, const std::vector<WordFeatures>& ws) const;
    void Backprop(const std::vector<WordFeatures>& ws, Label truth, const double* probabilities);

  public:
    BagOfWords(size_t in_sz, size_t out_sz);
    BagOfWords();

    const std::vector<std::vector<double>>& weights() const { return word_weight_; }
    const std::vector<double>& weights(size_t label) const { return word_weight_[label]; }
    double weight(size_t label, size_t w) const { return word_weight_[label][w]; }

    std::string Serialize() const;
    static BagOfWords FromSerialized(std::istream& file);

    double ComputeNLL(double* probas) const;
    Label ComputeClass(const std::vector<WordFeatures>& ws, double* probabilities) const;

    int Train(const Document& doc);

    void ResizeInput(size_t in);
    void ResizeOutput(size_t out);
};

