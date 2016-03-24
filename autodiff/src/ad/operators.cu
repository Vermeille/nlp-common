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

static void AddBackprop(Var& val, Var* lhs, Var* rhs) {
    cuAddBackprop bp {
        lhs->derivative().data().Get(),
        rhs->derivative().data().Get(),
        val.derivative().data().Get()
    };
    cuda::RunKernel(bp, val.value().size());
}

Var operator+(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(v1.value() + v2.value(), v1, v2, AddBackprop);
}

static void SubBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative();
    rhs->derivative() -= val.derivative();
}

Var operator-(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(v1.value() - v2.value(), v1, v2, SubBackprop);
}

Var operator-(float a, const Var& v1) {
    Matrix coeff(v1.value().rows(), v1.value().cols());
    coeff.SetConstant(a);
    Var coeff_var = v1.graph()->CreateParam(std::move(coeff));
    return v1.graph()->CreateNode(
            a - v1.value(), coeff_var, v1, SubBackprop);
}

Var operator-(const Var& v1, float a) {
    Matrix coeff(v1.value().rows(), v1.value().cols());
    coeff.SetConstant(a);
    Var coeff_var = v1.graph()->CreateParam(std::move(coeff));
    return v1.graph()->CreateNode(
            v1.value() - a, v1, coeff_var, SubBackprop);
}

static void MulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += MulNT(val.derivative(), rhs->value());
    rhs->derivative() += MulTN(lhs->value(), val.derivative());
}

static void CoeffMulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative() * rhs->value().CudaRead(0, 0);
    rhs->derivative() += MulTN(val.derivative(), lhs->value());
}

Var operator*(const Var& v1, const Var& v2) {
    if (v2.value().rows() == 1 && v2.value().cols() == 1) {
        return v2 * v1;
    }

    if (v1.value().rows() == 1 && v1.value().cols() == 1) {
        return v1.graph()->CreateNode(
                v2.value() * v1.value().CudaRead(0, 0), v2, v1, CoeffMulBackprop);
    }

    return v1.graph()->CreateNode(v1.value() * v2.value(), v1, v2, MulBackprop);
}

Var operator*(float a, const Var& v1) {
    Matrix coeff(1, 1);
    coeff.SetConstant(a);
    Var coeff_var = v1.graph()->CreateParam(std::move(coeff));
    return v1.graph()->CreateNode(v1.value() * a, v1, coeff_var, CoeffMulBackprop);
}

Var operator*(const Var& v1, float a) {
    Matrix coeff(1, 1);
    coeff.SetConstant(a);
    Var coeff_var = v1.graph()->CreateParam(std::move(coeff));
    return v1.graph()->CreateNode(v1.value() * a, v1, coeff_var, CoeffMulBackprop);
}

static void ReluBackprop(Var& val, Var* lhs, Var*) {
    Matrix relugrad(val.value().rows(), val.value().cols());
    auto g = cuda::Array(lhs->derivative().data());
    auto x = cuda::Array(lhs->value().data());
    auto dx = cuda::Array(val.derivative().data());
    cuda::RunKernel(cuda::Seq(g += (x > cuda::Value(0)) * dx), val.value().size());
}

Var Relu(const Var& v1) {
    Matrix res(v1.value().rows(), v1.value().cols());
    cuda::Relu(res.data(), v1.value().data(), res.size());
    return v1.graph()->CreateNode(
            std::move(res), v1, no_operand, ReluBackprop);
}

static void EltSquareBackprop(Var& val, Var* lhs, Var*) {
    auto dx = cuda::Array(lhs->derivative().data());
    auto dprev = cuda::Array(val.derivative().data());
    auto x = cuda::Array(lhs->value().data());
    cuda::RunKernel(cuda::Seq(dx += cuda::Value(2) * dprev * x), val.value().size());
}

Var EltSquare(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value() ^ v1.value(), v1, no_operand, EltSquareBackprop);
}

static void EltwiseMulBackprop(Var& val, Var* lhs, Var* rhs) {
    auto _dlhs = cuda::Array(lhs->derivative().data());
    auto _lhs = cuda::Array(lhs->value().data());
    auto _drhs = cuda::Array(rhs->derivative().data());
    auto _rhs = cuda::Array(rhs->value().data());
    auto _dprev = cuda::Array(val.derivative().data());
    cuda::RunKernel(cuda::Seq(
            _dlhs += _rhs * _dprev,
            _drhs += _lhs * _dprev),
        val.value().size());
}

Var operator^(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(
            v1.value() ^ v2.value(), v1, v2, EltwiseMulBackprop);
}

static void LogBackprop(Var& val, Var* lhs, Var*) {
    auto da = cuda::Array(lhs->derivative().data());
    auto dx = cuda::Array(val.derivative().data());
    auto a = cuda::Array(lhs->value().data());
    cuda::RunKernel(cuda::Seq(da += dx / a), val.derivative().size());
}

Var Log(const Var& val) {
    Matrix res(val.value().rows(), val.value().cols());
    auto r = cuda::Array(res.data());
    auto x = cuda::Array(val.value().data());
    cuda::RunKernel(cuda::Seq(r = cuda::log(x)), res.size());
    return val.graph()->CreateNode(std::move(res), val, no_operand, LogBackprop);
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

static void SoftmaxLossBackprop(Var& val, Var* lhs, Var* rhs) {
    Matrix softmaxed = Softmax(lhs->value());
    auto p = cuda::Array(softmaxed.data());
    auto dx = cuda::Array(lhs->derivative().data());
    auto y = cuda::Array(rhs->value().data());
    auto dprev = cuda::Array(val.derivative().data());
    cuda::RunKernel(cuda::Seq(dx += (p - y) * dprev), val.value().size());
}

Var SoftmaxLoss(Var h, Var target) {
    auto res = Softmax(h.value());
    auto r = cuda::Array(res.data());
    auto y = cuda::Array(target.value().data());
    cuda::RunKernel(cuda::Seq(r = cuda::Value(0) - y * cuda::log(r)), res.size());
    return h.graph()->CreateNode(std::move(res), h, target, SoftmaxLossBackprop);
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

static void SoftmaxBackprop(Var& val, Var* lhs, Var*) {
    cuSoftmaxBackprop bp {
        lhs->derivative().data().Get(),
        val.derivative().data().Get(),
        val.value().data().Get(),
        val.value().size()
    };
    cuda::RunKernel(bp, val.value().size());
}

Var Softmax(const Var& x) {
    return x.graph()->CreateNode(
            Softmax(x.value()), x, no_operand, SoftmaxBackprop);
}

static void SigmoidBackprop(Var& val, Var* lhs, Var*) {
    auto a = cuda::Array(val.value().data());
    auto dprev = cuda::Array(val.derivative().data());
    auto dlhs = cuda::Array(lhs->derivative().data());
    cuda::RunKernel(cuda::Seq(
            dlhs += dprev * (a * (cuda::Value(1.0) - a))),
        val.value().size());
}

Var Sigmoid(const Var& x) {
    Matrix res(x.value().rows(), x.value().cols());
    auto lhs = cuda::Array(x.value().data());
    auto r = cuda::Array(res.data());
    cuda::RunKernel(cuda::Seq(
            r = cuda::Value(1) / (cuda::Value(1) + exp(cuda::Value(0) - lhs))),
        x.value().size());
    return x.graph()->CreateNode(std::move(res), x, no_operand, SigmoidBackprop);
}

static void SumBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative() += val.derivative().CudaRead(0, 0);
}

Var Sum(const Var& a) {
    Matrix res(1, 1);
    res.CudaWrite(0, 0, a.value().sum());
    return a.graph()->CreateNode(std::move(res), a, no_operand, SumBackprop);
}

static void MeanBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative() +=
        val.derivative().CudaRead(0, 0) / val.value().size();
}

Var Mean(const Var& a) {
    Matrix res(1, 1);
    res.CudaWrite(0, 0, a.value().sum() / a.value().size());
    return a.graph()->CreateNode(std::move(res), a, no_operand, MeanBackprop);
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

    __device__
    inline
    void operator()(size_t i) {
        float denom = coshf(2 * val_[i]) + 1;
        denom = denom * denom;
        float num = coshf(val_[i]);
        num = 4 * num * num;
        dlhs_[i] = (num / denom) * dprev_[i];
    }
};

static void TanhBackprop(Var& val, Var* lhs, Var*) {
    TanhGrad tg = {
        .dlhs_ = lhs->derivative().data().Get(),
        .val_ = val.value().data().Get(),
        .dprev_ = val.derivative().data().Get()
    };

    cuda::RunKernel(tg, val.value().size());
}

Var Tanh(const Var& val) {
    Matrix res(val.value().rows(), val.value().cols());
    auto r = cuda::Array(res.data());
    auto x = cuda::Array(val.value().data());
    cuda::RunKernel(cuda::Seq(r = cuda::tanh(x)), res.size());
    return val.graph()->CreateNode(std::move(res), val, no_operand, TanhBackprop);
}

static void ColAppendBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative().block(0, 0, lhs->derivative().rows(), 1);
    rhs->derivative() += val.derivative().block(
            lhs->derivative().rows(), 0, rhs->derivative().rows(), 1);
}

Var ColAppend(Var x, Var y) {
    if (x.value().cols() != 1 || y.value().cols() != 1) {
        throw std::runtime_error("cannot append not-a-column-vectors");
    }

    Matrix cated(x.value().rows() + y.value().rows(), 1);
    cudaMemcpy(
            cated.data().Get(),
            x.value().data().Get(),
            sizeof (float) * x.value().rows(),
            cudaMemcpyDeviceToDevice);
    cudaMemcpy(
            cated.data().Get() + x.value().rows(),
            y.value().data().Get(),
            sizeof (float) * y.value().rows(),
            cudaMemcpyDeviceToDevice);
    return x.graph()->CreateNode(std::move(cated), x, y, ColAppendBackprop);
}

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

}
