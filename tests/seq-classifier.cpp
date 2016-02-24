#include <iostream>
#include <fstream>

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/sequence-classifier.h>

Document Parse(const std::string& str, NGramMaker& ngram, LabelSet& ls) {
    std::ifstream dataset(str);
    std::string line;
    Document doc;
    while (std::getline(dataset, line)) {
        size_t pipe = line.find('|');
        std::string data(line, 0, pipe - 1);
        std::string label(line, pipe + 2, line.size());

        std::vector<WordFeatures> toks = Tokenizer::FR(data);
        ngram.Learn(toks);
        doc.examples.push_back(TrainingExample{toks, ls.GetLabel(label)});
    }
    return doc;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>\n";
        return EXIT_FAILURE;
    }

    NGramMaker ngram;
    SequenceClassifier seq(0, 20, 20, 0);
    LabelSet ls;
    Document doc = Parse(argv[1], ngram, ls);

    seq.ResizeInput(ngram.dict().size());
    seq.ResizeOutput(ls.size());

    std::cout << "Training...\n";
    for (int i = 0; i < 50; ++i) {
        std::cout << seq.Train(doc) << "% accuracy" << std::endl;
    }

    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::FR(std::string(line));
        ngram.Annotate(toks);
        auto prediction = seq.ComputeClass(toks);

        for (size_t l = 0; l < ls.size(); ++l) {
            std::cout << ls.GetString(l) << ": " << prediction(l, 0) << "\n";
        }
    }

    return 0;
}
