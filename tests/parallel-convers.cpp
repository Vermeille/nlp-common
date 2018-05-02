#include <iostream>
#include <fstream>
#include <thread>

#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

#include <nlp/dict.h>
#include <nlp/tokenizer.h>
#include <nlp/conversation-layered.h>

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
std::vector<int> Answer(Params& params, const std::vector<WordFeatures>& in, int max, int space) {
    using namespace ad;
    std::vector<int> answer;
    ad::ComputationGraph g;
    Conversation conv(g, params, space);


    Var decoded = Softmax((1 / temp) * conv.Step(g, 0));
    for (auto& wf : in) {
        answer.push_back(wf.idx);
        decoded = Softmax((1 / temp) * conv.Step(g, wf.idx));
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
    const size_t mb_size = 2000;
    size_t ds_partition_size = doc.examples.size() / 6;
    opt::ParallelMBLocks locks;
#if 0
    train::IterativeSequenceTaggerTrainer<opt::Minibatch<opt::Adagrad>>
        trainer(opt::Minibatch<opt::Adagrad>(1024, opt::Adagrad()));
#else
    /*
    train::IterativeSequenceTaggerTrainer<opt::SGD> trainer(0.1);
    //*/
    //*/
#endif
    auto parallel_fun = [&](int tid) {
        size_t nb = 0;
        /*
        train::WholeSequenceTaggerTrainer<opt::ParallelMinibatch<opt::Momentum>>
            trainer(opt::ParallelMinibatch<opt::Momentum>(mb_size, 6, locks, opt::Momentum(0.001, 0.9)));
        //*/

        train::WholeSequenceTaggerTrainer
            trainer(new opt::ParallelMinibatch(mb_size, 6, locks, new opt::Adagrad(0.1)));
        for (int i = 0; i < 5000; ++i) {
            double cost = 0;
            for (int j = ds_partition_size * tid; j < ds_partition_size * (tid + 1); ++j) {
                auto& cur = doc.examples[j];
                std::vector<WordFeatures> sentence;
                sentence.emplace_back("|");
                sentence.back().idx = 0;
                sentence.insert(sentence.end(), cur.inputs.begin(), cur.inputs.end());

                std::vector<WordFeatures> target;
                target.insert(target.end(), cur.inputs.begin(), cur.inputs.end());
                target.emplace_back("|");
                target.back().idx = 0;

                cost += trainer.Step(sentence, target, [&](ComputationGraph& g) {
                        return Conversation(g, params, dict.GetWordIdOrUnk(" "));
                        });

                ++nb;

                if (nb % 100 == 0 && nb) {
                    std::vector<WordFeatures> test;
                    test.insert(test.end(), cur.inputs.begin(),
                            cur.inputs.begin() + std::min(5ul, cur.inputs.size()));
                    auto example = Answer(params, test, 80, dict.GetWordIdOrUnk(" "));
                    std::cout << "(" << tid << ") ";
                    for (int letter : example) {
                        std::cout << dict.WordFromId(letter);
                    }
                    std::cout << "\n";
                }
                if (nb % mb_size == 0 && nb) {
                    std::cout << nb << ": " << " (" << (cost / mb_size) << ") |";
                    cost = 0;
                }
            }
        }
    };
    std::thread t1(parallel_fun, 0);
    std::thread t2(parallel_fun, 1);
    std::thread t3(parallel_fun, 2);
    std::thread t4(parallel_fun, 3);
    std::thread t5(parallel_fun, 4);
    std::thread t6(parallel_fun, 5);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
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
    for (int i = 0; i < ngram.dict().size(); ++i) {
        std::cout << i << ":" << ngram.dict().WordFromId(i) << " ";
        if (i % 10 == 0 && i) {
            std::cout << std::endl;
        }
    }

    ConversationParams seq(ngram.dict().size(), 64, 32);

    std::cout << "Training...\n";
    Train(seq, doc, ngram.dict());

    char* line;
    while ((line = readline("> "))) {
        add_history(line);

        std::vector<WordFeatures> toks = Tokenizer::CharLevel(std::string(line));
        ngram.Annotate(toks);
        auto prediction = Answer(seq, toks, 100, ngram.dict().GetWordIdOrUnk(" "));

        for (auto p : prediction) {
            std::cout << ngram.dict().WordFromId(p);
        }
        std::cout << "\n";
    }

    return 0;
}
