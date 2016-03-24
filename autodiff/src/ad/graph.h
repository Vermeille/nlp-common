#pragma once

#include <list>
#include <unordered_map>
#include <memory>
#include <set>
#include <vector>

#include "optimizer.h"
#include "initializers.h"

namespace ad {

class Var;
class ComputationGraph;

using backward_t = void(*)(Var&, Var*, Var*);
void DoNothingBackprop(Var&, Var*, Var*);

class Param {
    private:
        Matrix value_;
        mutable size_t persistent_id_;
        static size_t next_id_;
    public:
        Param(size_t rows, size_t cols,
                const MatrixInitialization& init = Xavier());
        Param(Matrix&& val) : value_(std::move(val)), persistent_id_(0) { }
        const Matrix& value() const { return value_; }
        Matrix& value() { return value_; }
        size_t rows() const { return value_.rows(); }
        size_t cols() const { return value_.cols(); }
        size_t GetPersistentId() const {
            if (persistent_id_ != 0) {
                return persistent_id_;
            }
            ++next_id_;
            persistent_id_ = next_id_;
            return persistent_id_;
        }
};

class VarImpl {
    private:
        std::shared_ptr<Param> value_;
        Matrix derivative_;

        int lhs_;
        int rhs_;
        int id_;

        backward_t backward_;

        ComputationGraph* const graph_;
        bool learnable_;

    public:
        VarImpl(ComputationGraph* g,
                std::shared_ptr<Param> val,
                int my_id,
                int op1,
                int op2,
                const backward_t& bckwd,
                bool learnable)
            : value_(val),
            derivative_(val->value().rows(), val->value().cols()),
            lhs_(op1),
            rhs_(op2),
            id_(my_id),
            backward_(bckwd),
            graph_(g),
            learnable_(learnable) {
            derivative_.SetZero();
        }

        VarImpl(ComputationGraph* g,
                Matrix&& val,
                int my_id,
                int op1,
                int op2,
                const backward_t& bckwd,
                bool learnable)
            : value_(std::make_shared<Param>(std::move(val))),
            derivative_(val.rows(), val.cols()), lhs_(op1),
            rhs_(op2), id_(my_id), backward_(bckwd), graph_(g),
            learnable_(learnable) {
            derivative_.SetZero();
        }
        ComputationGraph* graph() const { return graph_; }
        const Matrix& value() const { return value_->value();}
        Matrix& value() { return value_->value();}
        const Matrix& derivative() const { return derivative_;}
        Matrix& derivative() { return derivative_;}

        size_t persistent_id() const { return value_->GetPersistentId(); }
        bool IsLearnable() const { return learnable_; }

        void ClearDerivative() { derivative_.SetZero(); }

        void Backward(Var& self, Var* lhs, Var* rhs) {
            backward_(self, lhs, rhs);
        }

        void InitBackprop() { derivative_.SetOnes(); }

        int id() const { return id_; }
        int lhs() const { return lhs_; }
        int rhs() const { return rhs_; }
};

class Var {
    VarImpl* var_;
    public:
        explicit Var(VarImpl* var) : var_(var) {}
        Var(const Var&) = default;
        ComputationGraph* graph() const { return var_->graph(); }
        const Matrix& value() const { return var_->value();}
        Matrix& value() { return var_->value();}
        const Matrix& derivative() const { return var_->derivative();}
        Matrix& derivative() { return var_->derivative();}
        bool IsLearnable() const { return var_->IsLearnable(); }

        void Backward(Var* lhs, Var* rhs) {
            var_->Backward(*this, lhs, rhs);
        }
        size_t persistent_id() const { return var_->persistent_id(); }
        int id() const { return var_->id(); }
        int lhs() const { return var_->lhs(); }
        int rhs() const { return var_->rhs(); }

        bool operator<(const Var& o) const { return var_ < o.var_; }
};

extern const Var no_operand;

class ComputationGraph {
    std::vector<std::unique_ptr<VarImpl>> values_;

    public:
    Var CreateParam(std::shared_ptr<Param> val, bool learnable = false);
    Var CreateParam(Matrix&& val);
    Var CreateNode(
            Matrix&& val,
            const Var& lhs,
            const Var& rhs,
            backward_t bwd);
    void BackpropFrom(Var& x, double clip = 0);
    void ClearGrad();
    void ClearIntermediateGradientsFrom(Var x);
    void Update(Optimizer& opt);
    std::vector<Var> GetAllLearnableVars() const { // FIXME make me iterable
        std::vector<Var> vars;
        for (auto& v : values_) {
            if (v->IsLearnable()) {
                vars.emplace_back(v.get());
            }
        }
        return vars;
    }
};

}

