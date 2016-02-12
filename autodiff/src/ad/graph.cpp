#include "graph.h"

namespace ad {

const Var no_operand(
        new VarImpl(nullptr, Eigen::MatrixXd(1,1), -1, -1, -1, DoNothingBackprop, false));
void DoNothingBackprop(Var&, Var*, Var*) {}

Var ComputationGraph::CreateParam(
        std::shared_ptr<Eigen::MatrixXd> val, bool learnable) {
    int p_id = values_.size();

    values_.emplace_back(
            new VarImpl(this, val, p_id, -1, -1, DoNothingBackprop, learnable));
    return Var(values_.back().get());
}

Var ComputationGraph::CreateParam(const Eigen::MatrixXd& val) {
    return CreateParam(std::make_shared<Eigen::MatrixXd>(val));
}

Var ComputationGraph::CreateNode(
        const Eigen::MatrixXd& val,
        const Var& lhs,
        const Var& rhs,
        backward_t bwd) {
    int p_id = values_.size();

    values_.emplace_back(
            new VarImpl(this, val, p_id, lhs.id(), rhs.id(), *bwd, false));
    return Var(values_.back().get());
}

static void Clip(Eigen::MatrixXd& mat, double clip) {
    double* data = mat.data();
    for (int i = 0; i < mat.size(); ++i) {
        data[i] = data[i] > clip ? clip : data[i];
        data[i] = data[i] < -clip ? -clip : data[i];
    }
}

void ComputationGraph::BackpropFrom(Var& x, double clip) {
    int id = x.id();
    values_[id]->InitBackprop();
    for (int i = id; i >= 0; --i) {
        Var cur(values_[i].get());
        if (clip != 0) {
            Clip(cur.derivative(), clip);
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
            cur.derivative().setZero();
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
