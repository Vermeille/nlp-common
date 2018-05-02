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

class Operator {
    private:
        std::shared_ptr<Param> value_;
        Matrix derivative_;

        ComputationGraph* const graph_;
        bool learnable_;

    public:
        Operator(ComputationGraph* g,
                std::shared_ptr<Param> val,
                bool learnable = false)
            : value_(val),
            derivative_(val->value().rows(), val->value().cols()),
            graph_(g),
            learnable_(learnable) {
            derivative_.SetZero();
        }

        Operator(ComputationGraph* g,
                Matrix&& val,
                bool learnable = false)
            : value_(std::make_shared<Param>(std::move(val))),
            derivative_(val.rows(), val.cols()),
            graph_(g),
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

        virtual void Backward() = 0;

        void InitBackprop() { derivative_.SetOnes(); }
};

class Var {
    Operator* var_;
    public:
        explicit Var(Operator* var) : var_(var) {}
        Var(const Var&) = default;
        ComputationGraph* graph() const { return var_->graph(); }
        const Matrix& value() const { return var_->value();}
        Matrix& value() { return var_->value();}
        const Matrix& derivative() const { return var_->derivative();}
        Matrix& derivative() { return var_->derivative();}
        bool IsLearnable() const { return var_->IsLearnable(); }

        void Backward() {
            var_->Backward();
        }
        size_t persistent_id() const { return var_->persistent_id(); }

        bool operator<(const Var& o) const { return var_ < o.var_; }
};

extern const Var no_operand;

class ComputationGraph {
    std::vector<std::shared_ptr<Operator>> values_;

    public:
    Var CreateParam(std::shared_ptr<Param> val, bool learnable = false);
    Var CreateParam(Matrix&& val);
    Var CreateNode(std::shared_ptr<Operator> op);
    void Backprop(double clip = 0);
    void ClearGrad();
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

