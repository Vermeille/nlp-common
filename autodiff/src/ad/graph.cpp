#include "graph.h"
#include "helpers.h"

namespace ad {

size_t Param::next_id_;

Param::Param(size_t rows, size_t cols, const MatrixInitialization& init)
    : value_(rows, cols),
    persistent_id_(0) {
        init.Init(value_);
}
Var ComputationGraph::CreateParam(
        std::shared_ptr<Param> val, bool learnable) {
    struct LoadOperator : public Operator {
        LoadOperator(
                ComputationGraph* g,
                const std::shared_ptr<Param>& p,
                bool learnable)
            : Operator(g, p, learnable) {}

        virtual void Backward() override {}
    };
    values_.push_back(std::make_shared<LoadOperator>(this, val, learnable));
    return Var(values_.back().get());
}

Var ComputationGraph::CreateParam(Matrix&& val) {
    return CreateParam(std::make_shared<Param>(std::move(val)));
}

Var ComputationGraph::CreateNode(std::shared_ptr<Operator> op) {
    values_.push_back(op);
    return Var(values_.back().get());
}

void ComputationGraph::Backprop(double clip) {
    values_.back()->InitBackprop();
    for (int i = values_.size() - 1; i >= 0; --i) {
        Operator& op = *values_[i];
        if (clip != 0) {
            op.derivative().Clip(clip);
        }
        op.Backward();
    }
}

void ComputationGraph::ClearGrad() {
    for (auto& v : values_) {
        v->ClearDerivative();
    }
}

void ComputationGraph::Update(Optimizer& opt) {
    for (auto& vptr : values_) {
        Var v(vptr.get());
        if (v.IsLearnable()) {
            opt.Update(v);
        }
    }
}

} // ad
