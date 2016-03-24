#include "graph.h"
#include "helpers.h"

namespace ad {

const Var no_operand(
        new VarImpl(nullptr, Matrix(1,1), -1, -1, -1, DoNothingBackprop, false));
void DoNothingBackprop(Var&, Var*, Var*) {}

size_t Param::next_id_;

Param::Param(size_t rows, size_t cols, const MatrixInitialization& init)
    : value_(rows, cols),
    persistent_id_(0) {
        init.Init(value_);
}
Var ComputationGraph::CreateParam(
        std::shared_ptr<Param> val, bool learnable) {
    int p_id = values_.size();

    values_.emplace_back(
            new VarImpl(this, val, p_id, -1, -1, DoNothingBackprop, learnable));
    return Var(values_.back().get());
}

Var ComputationGraph::CreateParam(Matrix&& val) {
    return CreateParam(std::make_shared<Param>(std::move(val)));
}

Var ComputationGraph::CreateNode(
        Matrix&& val,
        const Var& lhs,
        const Var& rhs,
        backward_t bwd) {
    int p_id = values_.size();

    values_.emplace_back(
            new VarImpl(
                this, std::move(val), p_id, lhs.id(), rhs.id(), *bwd, false));
    return Var(values_.back().get());
}

void ComputationGraph::BackpropFrom(Var& x, double clip) {
    int id = x.id();
    values_[id]->InitBackprop();
    for (int i = id; i >= 0; --i) {
        Var cur(values_[i].get());
        if (clip != 0) {
            cur.derivative().Clip(clip);
        }
        Var nullvar(nullptr);
        if (cur.lhs() == -1) {
            cur.Backward(nullptr, nullptr);
        } else if (cur.rhs() == -1) {
            Var a(values_[cur.lhs()].get());
            cur.Backward(&a, nullptr);
        } else {
            Var a(values_[cur.lhs()].get());
            Var b(values_[cur.rhs()].get());
            cur.Backward(&a, &b);
        }
    }
}

void ComputationGraph::ClearGrad() {
    for (auto& v : values_) {
        v->ClearDerivative();
    }
}

void ComputationGraph::ClearIntermediateGradientsFrom(Var x) {
    int id = x.id();
    for (int i = id; i >= 0; --i) {
        Var cur(values_[i].get());
        if (cur.lhs() != -1) {
            cur.derivative().SetZero();
        }
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
