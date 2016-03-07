#include <iostream>
#include <fstream>

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/bow.h>

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

double Test(BagOfWords& bow, const Document& doc) {
    int nb_correct = 0;
    int nb_tokens = 0;

    for (auto& ex : doc.examples) {
        ad::ComputationGraph g;
        auto h = bow.Step(g, ex.inputs);

        Label predicted = ad::utils::OneHotVectorDecode(h.value());
        nb_correct += predicted == ex.output ? 1 : 0;
        ++nb_tokens;
    }
    return nb_correct * 100 / nb_tokens;
}

void Train(BagOfWords& bow, const Document& doc) {
    ad::train::FeedForwardTrainer trainer(new ad::opt::Adagrad());
    for (int i = 0; i < 5; ++i) {
        double nll = 0;
        for (auto& ex : doc.examples) {
            nll += trainer.Step(ex.inputs, ex.output,
                    [&](ad::ComputationGraph&) {
                    return bow;
                });
        }
        std::cout << "cost: " << nll << "\n";
        std::cout << Test(bow, doc) << "% accuracy\n";
    }
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>\n";
        return EXIT_FAILURE;
    }

    NGramMaker ngram;
    LabelSet ls;
    Document doc = Parse(argv[1], ngram, ls);

    BagOfWords bow(ngram.dict().size(), ls.size());

    std::cout << "Training...\n";
    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::FR(std::string(line));
        ngram.Annotate(toks);
        auto prediction = bow.ComputeClass(toks);

        for (size_t l = 0; l < ls.size(); ++l) {
            std::cout << ls.GetString(l) << ": " << prediction(l, 0) << "\n";
        }
    }

    return 0;
}
