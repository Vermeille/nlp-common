#include <iostream>
#include <fstream>

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/conversation-mrnn.h>

const double temp = 1;

static int Sample(const Eigen::MatrixXd& probs) {
    double r = ad::utils::RandomRange(0, 1);
    double acc = 0;
    for (int i = 0; i < probs.rows(); ++i) {
        acc += probs(i, 0);
        if (acc > r) {
            return i;
        }
    }
    return probs.rows() - 1;
}

Document Parse(const std::string& str, NGramMaker& ngram) {
    std::ifstream dataset(str);
    std::string line;
    Document doc;
    while (std::getline(dataset, line)) {
        std::vector<WordFeatures> toks = Tokenizer::CharLevel(line);
        ngram.Learn(toks);
        doc.examples.push_back(TrainingExample{toks, 0});
    }

    return doc;
}

template <class Params>
std::vector<int> Answer(Params& params, const std::vector<WordFeatures>& in, int max) {
    using namespace ad;
    std::vector<int> answer;
    ad::ComputationGraph g;
    Conversation conv(g, params);

    Var decoded(nullptr);
    for (auto& wf : in) {
        decoded = Softmax((1 / temp) * conv.Step(g, wf));
        answer.push_back(wf.idx);
    }

    answer.push_back(Sample(decoded.value()));
    for (int j = 0; j < max - 2 && answer.back() != 0; ++j) {
        size_t word = answer.back();
        decoded = Softmax((1 / temp) * conv.Step(g, word));
        answer.push_back(Sample(decoded.value()));
    }
    return answer;
}

template <class Params>
double Train(Params& params, const Document& doc, const Dictionnary& dict) {
    using namespace ad;
    size_t nb = 0;

    train::IterativeSequenceTaggerTrainer<opt::SGD>
        trainer(opt::SGD(0.1));

    for (auto& cur : doc.examples) {
        std::vector<WordFeatures> sentence;
        sentence.insert(sentence.end(), cur.inputs.begin() +1, cur.inputs.end());
        sentence.emplace_back("|");
        sentence.back().idx = 0;

        double cost = trainer.Step(cur.inputs, sentence, [&](ComputationGraph& g) {
                return Conversation(g, params);
            });

        ++nb;

        if (nb % 20 == 0) {
            std::cout << nb << ": " << " (" << cost << ") |";
            std::vector<WordFeatures> test;
            test.insert(test.end(), cur.inputs.begin(),
                    cur.inputs.begin() + std::min(5ul, cur.inputs.size()));
            auto example = Answer(params, test, 80);
            for (int letter : example) {
                std::cout << dict.WordFromId(letter);
            }
            std::cout << "\n";
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset>\n";
        return EXIT_FAILURE;
    }

    NGramMaker ngram;
    Document doc = Parse(argv[1], ngram);

    std::cout << "VOCAB SIZE: " << ngram.dict().size() << "\n";

    ConversationParams seq(ngram.dict().size(), 50);

    std::cout << "Training...\n";
    for (int i = 0; i < 50; ++i) {
        Train(seq, doc, ngram.dict());
        std::cout << i << " done.\n";
    }

    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::CharLevel(std::string(line));
        ngram.Annotate(toks);
        auto prediction = Answer(seq, toks, 100);

        for (auto p : prediction) {
            std::cout << ngram.dict().WordFromId(p);
        }
        std::cout << "\n";
    }

    return 0;
}
