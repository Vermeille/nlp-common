#pragma once

namespace ad {
namespace opt {

class Minibatch : public Optimizer {
    private:
        const int batch_size_;
        std::unique_ptr<Optimizer> opt_;
        std::map<size_t, std::pair<int, Matrix>> grad_mean_;

        std::pair<int, Matrix>& SetCurrentGradient(Var v) {
            auto grad = grad_mean_.find(v.persistent_id());
            if (grad == grad_mean_.end()) {
                grad = grad_mean_.insert(
                        std::make_pair(
                            v.persistent_id(),
                            std::make_pair(
                                1,
                                (1.0 / batch_size_) * v.derivative()
                            )
                        )
                    ).first;
            } else {
                ++grad->second.first;
                grad->second.second += (1.0 / batch_size_) * v.derivative();
            }
            return grad->second;
        }


    public:
        Minibatch(int size, Optimizer* opt) : batch_size_(size), opt_(opt) {}

        virtual void Update(Var& v) {
            auto& cur = SetCurrentGradient(v);
            if (cur.first != batch_size_) {
                return;
            }
            cur.first = 0;
            v.derivative() = std::move(cur.second);
            cur.second.SetZero();
            opt_->Update(v);
        }
};

} // opt
} // ad
