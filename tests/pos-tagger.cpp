#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/bow.h>
#include <nlp/sequence-tagger.h>

Document Parse(const std::string& str, NGramMaker& ngram, LabelSet& ls) {
    std::ifstream dataset(str);
    std::string word;
    std::string pos;
    std::string line;
    Document doc;
    while (std::getline(dataset, line)) {
        std::istringstream l(line);
        std::vector<WordFeatures> toks;

        l >> word;
        l >> pos;
        while (l) {
            toks.emplace_back(word);
            toks.back().pos = ls.GetLabel(pos);
            l >> word;
            l >> pos;
        }

        ngram.Learn(toks);
        doc.examples.push_back(TrainingExample{toks, 0});
    }
    return doc;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>\n";
        return EXIT_FAILURE;
    }

    NGramMaker ngram;
    LabelSet ls;
    SequenceTagger tagger(200, 2);
    Document doc = Parse(argv[1], ngram, ls);

    tagger.ResizeInput(ngram.dict().size());
    tagger.ResizeOutput(ls.size());

    std::cout << "Training...\n";
    for (int i = 0; i < 1; ++i) {
        std::cout << tagger.Train(doc) << "% accuracy" << std::endl;
    }

    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::FR(std::string(line));
        ngram.Annotate(toks);
        tagger.Compute(toks);

        for (auto& w : toks) {
            std::cout << w.str << ": " << ls.GetString(w.pos) << "\n";
        }
    }

    return 0;
}
