#include <cassert>

#include "kernels/kernels.h"
#include "operators.h"

using cuptr = ::cuda::Ptr<float>;

namespace ad {

struct cuAddBackprop {
    float* lhs;
    float* rhs;
    float* val;

    CUDA_CALLABLE
    inline
    void operator()(size_t i) {
        float tmp = val[i];
        lhs[i] += tmp;
        rhs[i] += tmp;
    }
};

Var operator+(const Var& v1, const Var& v2) {
    class AddOperator : public Operator {
        Var lhs_;
        Var rhs_;

        public:
            AddOperator(Var lhs, Var rhs)
                : Operator(
                        lhs.graph(),
                        lhs.value() + rhs.value()),
                lhs_(lhs),
                rhs_(rhs) {
            }

            virtual void Backward() override {
                cuAddBackprop bp {
                    lhs_.derivative().data().Get(),
                    rhs_.derivative().data().Get(),
                    derivative().data().Get()
                };
                cuda::RunKernel(bp, value().size());
            }
    };
    return v1.graph()->CreateNode(std::make_shared<AddOperator>(v1, v2));
}

Var operator-(const Var& v1, const Var& v2) {
    class SubOperator : public Operator {
        Var lhs_;
        Var rhs_;

        public:
            SubOperator(Var lhs, Var rhs)
                : Operator(
                        lhs.graph(),
                        lhs.value() - rhs.value()),
                lhs_(lhs),
                rhs_(rhs) {
            }

            virtual void Backward() override {
                lhs_.derivative() += derivative();
                rhs_.derivative() -= derivative();
            }
    };
    return v1.graph()->CreateNode(std::make_shared<SubOperator>(v1, v2));
}

Var operator-(float a, const Var& v1) {
    class SubCoeffOperator : public Operator {
        float lhs_;
        Var rhs_;

        public:
            SubCoeffOperator(float lhs, Var rhs)
                : Operator(
                        rhs.graph(),
                        lhs - rhs.value()),
                lhs_(lhs),
                rhs_(rhs) {
            }

            virtual void Backward() override {
                rhs_.derivative() -= derivative();
            }
    };
    return v1.graph()->CreateNode(std::make_shared<SubCoeffOperator>(a, v1));
}

Var operator-(const Var& v1, float a) {
    class SubCoeffOperator : public Operator {
        Var lhs_;
        float rhs_;

        public:
            SubCoeffOperator(Var lhs, float rhs)
                : Operator(
                        lhs.graph(),
                        lhs.value() - rhs),
                lhs_(lhs),
                rhs_(rhs) {
            }

            virtual void Backward() override {
                lhs_.derivative() += derivative();
            }
    };
    return v1.graph()->CreateNode(std::make_shared<SubCoeffOperator>(v1, a));
}

Var operator*(const Var& v1, const Var& v2) {
    if (v2.value().rows() == 1 && v2.value().cols() == 1) {
        return v2 * v1;
    }

    if (v1.value().rows() == 1 && v1.value().cols() == 1) {
        class MulCoeffOperator : public Operator {
            Var lhs_;
            Var rhs_;

            public:
                MulCoeffOperator(Var lhs, Var rhs)
                    : Operator(
                            lhs.graph(),
                            lhs.value() * rhs.value().CudaRead(0, 0)),
                    lhs_(lhs),
                    rhs_(rhs) {
                }

                virtual void Backward() override {
                    lhs_.derivative() += derivative() * rhs_.value().CudaRead(0, 0);
                    rhs_.derivative() += MulTN(derivative(), lhs_.value());
                }
        };
        return v1.graph()->CreateNode(
                std::make_shared<MulCoeffOperator>(v2, v1));
    }

    class MulOperator : public Operator {
        Var lhs_;
        Var rhs_;

        public:
            MulOperator(Var lhs, Var rhs)
                : Operator(
                        lhs.graph(),
                        lhs.value() * rhs.value()),
                lhs_(lhs),
                rhs_(rhs) {
            }

            virtual void Backward() override {
                lhs_.derivative() += MulNT(derivative(), rhs_.value());
                rhs_.derivative() += MulTN(lhs_.value(), derivative());
            }
    };
    return v1.graph()->CreateNode(std::make_shared<MulOperator>(v1, v2));
}

class MulCoeffWithCteOperator : public Operator {
    Var lhs_;
    float rhs_;

    public:
        MulCoeffWithCteOperator(Var lhs, float rhs)
            : Operator(lhs.graph(), lhs.value() * rhs),
            lhs_(lhs),
            rhs_(rhs) {
        }

        virtual void Backward() override {
            lhs_.derivative() += derivative() * rhs_;
        }
};

Var operator*(float a, const Var& v1) {
    return v1.graph()->CreateNode(
            std::make_shared<MulCoeffWithCteOperator>(v1, a));
}

Var operator*(const Var& v1, float a) {
    return v1.graph()->CreateNode(
            std::make_shared<MulCoeffWithCteOperator>(v1, a));
}

Var Relu(const Var& v1) {
    class ReluOperator : public Operator {
        Var lhs_;

        public:
            ReluOperator(Var lhs)
                : Operator(
                        lhs.graph(),
                        Matrix(lhs.value().rows(), lhs.value().cols())),
                lhs_(lhs) {
                cuda::Relu(value().data(), lhs_.value().data(), value().size());
            }

            virtual void Backward() override {
                auto g = cuda::Array(lhs_.derivative().data());
                auto x = cuda::Array(lhs_.value().data());
                auto dx = cuda::Array(derivative().data());
                cuda::RunKernel(cuda::Seq(
                            g += (x > cuda::Value(0)) * dx),
                        value().size());
            }
    };
    return v1.graph()->CreateNode(std::make_shared<ReluOperator>(v1));
}

Var EltSquare(const Var& v1) {
    class EltSquare : public Operator {
        Var lhs_;

        public:
            EltSquare(Var lhs)
                : Operator(lhs.graph(), lhs.value() ^ lhs.value()),
                lhs_(lhs) {
            }

            virtual void Backward() override {
                auto dx = cuda::Array(lhs_.derivative().data());
                auto dprev = cuda::Array(derivative().data());
                auto x = cuda::Array(lhs_.value().data());
                cuda::RunKernel(cuda::Seq(
                            dx += cuda::Value(2) * dprev * x),
                        value().size());
            }
    };
    return v1.graph()->CreateNode(std::make_shared<EltSquare>(v1));
}

Var operator^(const Var& v1, const Var& v2) {
    class EltwiseMul : public Operator {
        Var lhs_;
        Var rhs_;

        public:
            EltwiseMul(Var lhs, Var rhs)
                : Operator(lhs.graph(), lhs.value() ^ rhs.value()),
                lhs_(lhs),
                rhs_(rhs) {
            }

            virtual void Backward() override {
                auto _dlhs = cuda::Array(lhs_.derivative().data());
                auto _lhs = cuda::Array(lhs_.value().data());
                auto _drhs = cuda::Array(rhs_.derivative().data());
                auto _rhs = cuda::Array(rhs_.value().data());
                auto _dprev = cuda::Array(derivative().data());
                cuda::RunKernel(cuda::Seq(
                            _dlhs += _rhs * _dprev,
                            _drhs += _lhs * _dprev),
                        value().size());
            }
    };
    return v1.graph()->CreateNode(std::make_shared<EltwiseMul>(v1, v2));
}

Var Log(const Var& val) {
    class LogOperator : public Operator {
        Var lhs_;

        public:
            LogOperator(Var lhs)
                : Operator(lhs.graph(),
                        Matrix(lhs.value().rows(), lhs.value().cols())),
                lhs_(lhs) {
                auto r = cuda::Array(value().data());
                auto x = cuda::Array(lhs_.value().data());
                cuda::RunKernel(cuda::Seq(r = cuda::log(x)), value().size());
            }

            virtual void Backward() override {
                auto da = cuda::Array(lhs_.derivative().data());
                auto dx = cuda::Array(derivative().data());
                auto a = cuda::Array(lhs_.value().data());
                cuda::RunKernel(
                        cuda::Seq(da += dx / a),
                        derivative().size());
            }
    };
    return val.graph()->CreateNode(std::make_shared<LogOperator>(val));
}

Var CrossEntropy(const Var& h, const Var& y) {
    return 0 - Sum(y ^ Log(h));
}

static Matrix Softmax(const Matrix& x1) {
    Matrix res(x1.rows(), x1.cols());
    auto r = cuda::Array(res.data());
    auto x = cuda::Array(x1.data());
    cuda::RunKernel(cuda::Seq(r = cuda::exp(x)), x1.size());
    float total = res.sum();
    auto t = cuda::Value(total);
    cuda::RunKernel(cuda::Seq(r = r / t), res.size());
    return res;
}

Var SoftmaxLoss(Var h, Var target) {
    class SoftmaxLossOperator : public Operator {
        Var predicted_;
        Var target_;
        Matrix softmaxed_;

        public:
            SoftmaxLossOperator(Var h, Var t)
                : Operator(h.graph(),
                        Matrix(h.value().rows(), h.value().cols())),
                predicted_(h),
                target_(t),
                softmaxed_(Softmax(h.value())) {
                    auto r = cuda::Array(softmaxed_.data());
                    auto s = cuda::Array(value().data());
                    auto y = cuda::Array(target_.value().data());
                    cuda::RunKernel(
                            cuda::Seq(s = cuda::Value(0) - y * cuda::log(r)),
                            softmaxed_.size());
            }

            virtual void Backward() override {
                auto p = cuda::Array(softmaxed_.data());
                auto dx = cuda::Array(predicted_.derivative().data());
                auto y = cuda::Array(target_.value().data());
                auto dprev = cuda::Array(derivative().data());
                cuda::RunKernel(
                        cuda::Seq(dx += (p - y) * dprev),
                        value().size());
            }
    };
    return h.graph()->CreateNode(
            std::make_shared<SoftmaxLossOperator>(h, target));
}

struct cuSoftmaxBackprop {
    float* dlhs_;
    const float* dval_;
    const float* val_;
    size_t len_;

    CUDA_CALLABLE
    void operator()(size_t i) {
        float d = 0;
        for (size_t j = 0; j < len_; ++j) {
            d += val_[i] * (((i == j) ? 1 : 0) - val_[j]) * dval_[j];
        }
        dlhs_[i] += d;
    }
};

Var Softmax(const Var& x) {
    class SoftmaxOperator : public Operator {
        Var lhs_;

        public:
            SoftmaxOperator(Var lhs)
                : Operator(lhs.graph(), Softmax(lhs.value())),
                lhs_(lhs) {
            }

            virtual void Backward() override {
                cuSoftmaxBackprop bp {
                    lhs_.derivative().data().Get(),
                    derivative().data().Get(),
                    value().data().Get(),
                    value().size()
                };
                cuda::RunKernel(bp, value().size());
            }
    };
    return x.graph()->CreateNode(std::make_shared<SoftmaxOperator>(x));
}

Var Sigmoid(const Var& x) {
    class SigmoidOperator : public Operator {
        Var lhs_;

        public:
            SigmoidOperator(Var lhs)
                : Operator(
                        lhs.graph(),
                        Matrix(lhs.value().rows(), lhs.value().cols())),
                lhs_(lhs) {
                    auto culhs = cuda::Array(lhs_.value().data());
                    auto r = cuda::Array(value().data());
                    cuda::RunKernel(cuda::Seq(
                            r = cuda::Value(1) /
                                (cuda::Value(1) + exp(cuda::Value(0) - culhs))),
                        value().size());
            }

            virtual void Backward() override {
                auto a = cuda::Array(value().data());
                auto dprev = cuda::Array(derivative().data());
                auto dlhs = cuda::Array(lhs_.derivative().data());
                cuda::RunKernel(cuda::Seq(
                            dlhs += dprev * (a * (cuda::Value(1.0) - a))),
                        value().size());
            }
    };
    return x.graph()->CreateNode(std::make_shared<SigmoidOperator>(x));
}

Var Sum(const Var& a) {
    class SumOperator : public Operator {
        Var lhs_;

        public:
            SumOperator(Var lhs)
                : Operator(lhs.graph(), Matrix(1, 1)),
                lhs_(lhs) {
                    value().CudaWrite(0, 0, lhs.value().sum());
            }

            virtual void Backward() override {
                lhs_.derivative() += derivative().CudaRead(0, 0);
            }
    };
    return a.graph()->CreateNode(std::make_shared<SumOperator>(a));
}

Var Mean(const Var& a) {
    class MeanOperator : public Operator {
        Var lhs_;

        public:
            MeanOperator(Var lhs)
                : Operator(lhs.graph(), Matrix(1, 1)),
                lhs_(lhs) {
                    value().CudaWrite(0, 0,
                            lhs.value().sum() / lhs.value().size());
            }

            virtual void Backward() override {
                lhs_.derivative()
                    += derivative().CudaRead(0, 0) / lhs_.value().size();
            }
    };
    return a.graph()->CreateNode(std::make_shared<MeanOperator>(a));
}

Var MSE(const Var& h, const Var& y) {
    return Mean(EltSquare(h - y));
}

Var SSE(const Var& h, const Var& y) {
    return Sum(EltSquare(h - y));
}

struct TanhGrad {
    float* dlhs_;
    float* val_;
    float* dprev_;

    CUDA_CALLABLE
    inline
    void operator()(size_t i) {
        float denom = coshf(2 * val_[i]) + 1;
        denom = denom * denom;
        float num = coshf(val_[i]);
        num = 4 * num * num;
        dlhs_[i] = (num / denom) * dprev_[i];
    }
};

Var Tanh(const Var& val) {
    class TanhOperator : public Operator {
        Var lhs_;

        public:
            TanhOperator(Var lhs)
                : Operator(
                        lhs.graph(),
                        Matrix(lhs.value().rows(), lhs.value().cols())),
                lhs_(lhs) {
                    auto r = cuda::Array(value().data());
                    auto x = cuda::Array(lhs.value().data());
                    cuda::RunKernel(cuda::Seq(r = cuda::tanh(x)),
                            lhs.value().size());
            }

            virtual void Backward() override {
                TanhGrad tg = {
                    .dlhs_ = lhs_.derivative().data().Get(),
                    .val_ = value().data().Get(),
                    .dprev_ = derivative().data().Get()
                };

                cuda::RunKernel(tg, value().size());
            }
    };
    return val.graph()->CreateNode(std::make_shared<TanhOperator>(val));
}

Var ColAppend(Var x, Var y) {
    class ColAppendOperator : public Operator {
        Var lhs_;
        Var rhs_;

        public:
            ColAppendOperator(Var lhs, Var rhs)
                : Operator(
                        lhs.graph(),
                        Matrix(lhs.value().rows() + rhs.value().rows(), 1)),
                lhs_(lhs),
                rhs_(rhs) {
                    if (lhs.value().cols() != 1 || rhs.value().cols() != 1) {
                        throw std::runtime_error(
                                "cannot append not-a-column-vectors");
                    }

                    cudaMemcpy(
                            value().data().Get(),
                            lhs.value().data().Get(),
                            sizeof (float) * lhs.value().rows(),
                            cudaMemcpyDeviceToDevice);
                    cudaMemcpy(
                            value().data().Get() + lhs.value().rows(),
                            rhs.value().data().Get(),
                            sizeof (float) * rhs.value().rows(),
                            cudaMemcpyDeviceToDevice);
            }

            virtual void Backward() override {
                lhs_.derivative() += derivative().block(0, 0, lhs_.derivative().rows(), 1);
                rhs_.derivative() += derivative().block(
                        lhs_.derivative().rows(), 0, rhs_.derivative().rows(), 1);
            }
    };
    return x.graph()->CreateNode(std::make_shared<ColAppendOperator>(x, y));
}

#if 0
static void ColSplitBackprop(Var& val, Var* lhs, Var* params) {
    int from = params->value().CudaRead(0, 0);
    int len = params->value().CudaRead(1, 0);
    float one = 1;
    cublasSaxpy(
            ::cuda::g_cuhandle.get(),
            len,
            &one,
            lhs->derivative().data().Get() + from,
            1,
            val.derivative().data().Get(),
            1);
}

Var ColSplit(Var x, int from, int len) {
    if (x.value().cols() != 1) {
        throw std::runtime_error("cannot split not-a-column-vectors");
    }

    Matrix params(2, 1);
    params.CudaWrite(0, 0, from);
    params.CudaWrite(1, 0, len);
    return x.graph()->CreateNode(
            x.value().block(from, 0, len, 1),
            x,
            x.graph()->CreateParam(std::move(params)),
            ColSplitBackprop);
}

#endif
}
