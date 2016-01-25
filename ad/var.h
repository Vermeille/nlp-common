#pragma once
#include <memory>

#include "Eigen/Dense"

namespace ad {

class Val;

void DoNothing(Val&);

typedef void (*backprop_t)(Val&);

class Val {
    private:
        Eigen::MatrixXd a_;
        Eigen::MatrixXd da_;
        std::shared_ptr<Val> dep1_;
        std::shared_ptr<Val> dep2_;
        backprop_t backprop_;

    public:
        Val& dep1() { return *dep1_; }
        Val& dep2() { return *dep2_; }
        Val(size_t rows,
            size_t cols,
            backprop_t bp = DoNothing,
            std::shared_ptr<Val> dep1 = nullptr,
            std::shared_ptr<Val> dep2 = nullptr);

        Val(Eigen::MatrixXd a,
            backprop_t bp = DoNothing,
            std::shared_ptr<Val> dep1 = nullptr,
            std::shared_ptr<Val> dep2 = nullptr);

        void Backprop();
        void ClearGrad();
        void Resize(size_t rows, size_t cols);

        Eigen::MatrixXd& derivative() { return da_; }
        Eigen::MatrixXd& val() { return a_; }

};

enum class VarType {
    Input,
    Param
};

class Var {
    std::shared_ptr<Val> v_;
    bool learnable_;

    public:
        Var(size_t rows, size_t cols, VarType type = VarType::Input) :
                v_(std::make_shared<Val>(rows, cols)) {
            SetType(type);
        }

        void SetType(VarType ty) {
            learnable_ = (ty == VarType::Param);
        }

        bool IsLearnable() const { return learnable_; }
        Eigen::MatrixXd& val() { return v_->val(); }
        const Eigen::MatrixXd& val() const { return v_->val(); }
        Eigen::MatrixXd& derivative() { return v_->derivative(); }
        void Backprop() {
            v_->derivative().setOnes();
            v_->Backprop();
        }
        Var(std::shared_ptr<Val> v) : v_(v) {}
        std::shared_ptr<Val> matrix() const { return v_; }
        void ClearGrad() { v_->ClearGrad(); }
        void Resize(size_t rows, size_t cols) { v_->Resize(rows, cols); }
};

}
