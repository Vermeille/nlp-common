#include <iostream>
#include <fstream>
#include <fenv.h>

#include <thread>
#include <chrono>

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/autoencoder-w-bidir.h>

Document Parse(const std::string& str, NGramMaker& ngram) {
    std::ifstream dataset(str);
    std::string line;
    Document doc;
    while (std::getline(dataset, line)) {
        std::vector<WordFeatures> toks = Tokenizer::FR(line);
        ngram.Learn(toks);
        doc.examples.push_back(TrainingExample{toks, 0});
    }

    return doc;
}

std::vector<int> SupervisedDecode(
        const AutoEncoderFullParams& params,
        const std::vector<WordFeatures>& in,
        int space,
        int max) {
    using namespace ad;
    std::vector<int> answer;
    ad::ComputationGraph g;
    AutoEncoderFull autoenc(g, params, space);

    for (Var x : autoenc.SupervisedStep(g, in)) {
        answer.push_back(utils::OneHotVectorDecode(x.value()));
    }

    return answer;
}

std::vector<int> Decode(
        const AutoEncoderFullParams& params,
        const std::vector<WordFeatures>& in,
        int space,
        int max) {
    using namespace ad;
    std::vector<int> answer;
    ad::ComputationGraph g;
    AutoEncoderFull autoenc(g, params, space);

    for (Var x : autoenc.FreestyleStep(g, in)) {
        answer.push_back(utils::OneHotVectorDecode(x.value()));
        //std::cout << answer.back() << ": " << x.value().CudaRead(answer.back(), 0) << "\n";
    }

    return answer;
}

double Train(AutoEncoderFullParams& params, const Document& doc, const Dictionnary& dict) {
    using namespace ad;
    const size_t mb_size = 10;
    const size_t threads = 3;
    size_t ds_partition_size = 1000 / threads;

    size_t nb = 0;
    train::FeedForwardTrainer
        trainer(new opt::Minibatch(mb_size, new opt::Adam(0.001)));

    size_t nb_whole = 100;

    auto start = std::chrono::system_clock::now();
    while (1) {
    double loss = 0;
    for (size_t i = 0; i < doc.examples.size(); ++i) {
        auto& cur = doc.examples[i];
        if (cur.inputs.empty()) {
            continue;
        }

        {
            loss += trainer.Step(cur.inputs, cur.inputs, [&](ComputationGraph& g) {
                    return AutoEncoderFull(g, params, dict.GetWordIdOrUnk(" "));
                });
        }
        ++nb;

        if (nb % 40 == 0) {
            std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << "ms\n";
#if 1
            std::cout << "(" << loss << ")\n";
            loss = 0;

            for (auto& letter : cur.inputs) {
                std::cout << dict.WordFromId(letter.idx) << " ";
            }
            std::cout << "\n";

            auto example = SupervisedDecode(params, cur.inputs, dict.GetWordIdOrUnk(" "),
                    cur.inputs.size());
            for (int letter : example) {
                std::cout << dict.WordFromId(letter) << " ";
            }
            std::cout << "\n";
#endif
            start = std::chrono::system_clock::now();
        }
    }
    ++nb_whole;
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

    AutoEncoderFullParams encdec(ngram.dict().size(), 1024, 1024);

    std::cout << "Training...\n";
    for (int i = 0; i < 500; ++i) {
        Train(encdec, doc, ngram.dict());
        std::cout << i << " done.\n";
    }

    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::CharLevel(std::string(line));
        ngram.Annotate(toks);
        auto prediction = Decode(encdec, toks, ngram.dict().GetWordIdOrUnk(" "), 100);

        for (auto p : prediction) {
            std::cout << ngram.dict().WordFromId(p);
        }
        std::cout << "\n";
    }

    return 0;
}
