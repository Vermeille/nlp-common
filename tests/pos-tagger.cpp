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

void InteractiveShell(SequenceTaggerParams& tagger, NGramMaker& ngram, LabelSet& ls) {
    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::FR(std::string(line));
        if (toks.empty()) {
            continue;
        }
        ngram.Annotate(toks);
        tagger.Compute(toks);

        for (auto& w : toks) {
            std::cout << w.str << ": " << ls.GetString(w.pos) << "\n";
        }
    }
}

void Train(const std::string& dataset, const std::string& model) {
    NGramMaker ngram;
    LabelSet ls;
    Document doc = Parse(dataset, ngram, ls);

    SequenceTaggerParams tagger(ngram.dict().size(), ls.size());

    std::cout << tagger.Train(doc) << "% accuracy FINAL" << std::endl;

    std::ofstream out(model);
    out << ngram.Serialize();
    out << ls.Serialize();
    out << tagger.Serialize();

    InteractiveShell(tagger, ngram, ls);
}

void Interactive(const std::string& model) {
    std::ifstream in(model);
    std::cout << "Loading model...";
    std::cout.flush();
    NGramMaker ngram = NGramMaker::FromSerialized(in);
    LabelSet ls = LabelSet::FromSerialized(in);
    SequenceTaggerParams tagger = SequenceTaggerParams::FromSerialized(in);
    std::cout << "done\n";

    InteractiveShell(tagger, ngram, ls);
}

int main(int argc, char** argv) {
    if (!strcmp(argv[1], "--train") && argc == 4) {
        Train(argv[2], argv[3]);
        return EXIT_SUCCESS;
    }

    if (!strcmp(argv[1], "--interactive") && argc == 3) {
        Interactive(argv[2]);
        return EXIT_SUCCESS;
    }

    std::cerr << "Usage: " << argv[0] << " --train <dataset> <model>\n";
    std::cerr << "Usage: " << argv[0] << " --interactive <model>\n";
    return EXIT_FAILURE;
}
