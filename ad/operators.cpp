#include <cassert>

#include "operators.h"

namespace ad {

static void AddBackprop(Val& v) {
    v.dep1().derivative() += v.derivative();
    v.dep2().derivative() += v.derivative();
}

Var operator+(const Var& v1, const Var& v2) {
    return Var(std::make_shared<Val>(v1.val() + v2.val(),
                AddBackprop,
                v1.matrix(),
                v2.matrix()));
}

static void SubBackprop(Val& v) {
    v.dep1().derivative() += v.derivative();
    v.dep2().derivative() -= v.derivative();
}

Var operator-(const Var& v1, const Var& v2) {
    return Var(std::make_shared<Val>(v1.val() - v2.val(),
                SubBackprop,
                v1.matrix(),
                v2.matrix()));
}

static void MulBackprop(Val& v) {
    v.dep1().derivative() +=  v.derivative() * v.dep2().val().transpose();
    v.dep2().derivative() +=  v.dep1().val().transpose() * v.derivative();
}

Var operator*(const Var& v1, const Var& v2) {
    return Var(std::make_shared<Val>(v1.val() * v2.val(),
                MulBackprop,
                v1.matrix(),
                v2.matrix()));
}

static void ReluBackprop(Val& v) {
    double* da = v.dep1().derivative().data();
    double* a = v.dep1().val().data();
    double* db = v.derivative().data();
    for (int i = 0; i < v.dep1().derivative().size(); ++i) {
        da[i] += a[i] > 0 ? db[i] : 0;
    }
}

Var Relu(const Var& v1) {
    return Var(std::make_shared<Val>(v1.val().array().max(0),
                ReluBackprop,
                v1.matrix()));
}

static void SquareBackprop(Val& v) {
    v.dep1().derivative() += 2 * v.derivative() * v.dep1().val();
}

Var Square(const Var& v1) {
    return Var(std::make_shared<Val>(v1.val() * v1.val(),
                SquareBackprop,
                v1.matrix()));
}

static void EltwiseMulBackprop(Val& x) {
    x.dep1().derivative() = x.derivative().cwiseProduct(x.dep2().val());
    x.dep2().derivative() = x.derivative().cwiseProduct(x.dep1().val());
}

Var operator^(const Var& a, const Var& b) {
    return Var(std::make_shared<Val>(a.val().cwiseProduct(b.val()),
                EltwiseMulBackprop,
                a.matrix(),
                b.matrix()));
}

static void LogBackprop(Val& x) {
    double* da = x.dep1().val().data();
    double* dx = x.derivative().data();
    double* a = x.dep1().val().data();
    double log10 = log(10);
    for (int i = 0; i < x.val().size(); ++i) {
        da[i] = dx[i] * 1 / (a[i] * log10);
    }
}

Var Log(const Var& x) {
    Eigen::MatrixXd res(x.val().rows(), x.val().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.val().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = log(src_ptr[i]);
    }
    return Var(std::make_shared<Val>(res, LogBackprop, x.matrix()));
}

Var NLog(const Var& x) {
    Eigen::MatrixXd res(x.val().rows(), x.val().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.val().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = -log(src_ptr[i]);
    }
    return Var(std::make_shared<Val>(res, LogBackprop, x.matrix()));
}

Var MSE(const Var& h, const Var& y) {
    return Square(h - y);
}

Var CrossEntropy(const Var& h, const Var& y) {
    return y ^ NLog(h);
}

static void ExpBackprop(Val& x) {
    double* da = x.dep1().val().data();
    double* dx = x.derivative().data();
    double* a = x.dep1().val().data();
    for (int i = 0; i < x.val().size(); ++i) {
        da[i] = dx[i] * exp(a[i]);
    }
}

Var Exp(const Var& x) {
    Eigen::MatrixXd res(x.val().rows(), x.val().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.val().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = exp(src_ptr[i]);
    }
    return Var(std::make_shared<Val>(res, ExpBackprop, x.matrix()));
}

static void SoftmaxBackprop(Val& x) {
    const Eigen::MatrixXd a = x.dep1().val();
    x.dep1().derivative() += x.derivative()
        .cwiseProduct((a.array() * (1.0 - a.array())).matrix());
}

Var Softmax(const Var& x) {
    Eigen::MatrixXd res(x.val().rows(), x.val().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.val().data();
    double total = 0;
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = exp(src_ptr[i]);
        total += dst_ptr[i];
    }

    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] /= total;
    }
    return Var(std::make_shared<Val>(res, SoftmaxBackprop, x.matrix()));
}

Var Sigmoid(const Var& x) {
    Eigen::MatrixXd res(x.val().rows(), x.val().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.val().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = 1.0 / (1 + exp(-src_ptr[i]));
    }

    // Sigmoid derivative is the same as Softmax's
    return Var(std::make_shared<Val>(res, SoftmaxBackprop, x.matrix()));
}

}
