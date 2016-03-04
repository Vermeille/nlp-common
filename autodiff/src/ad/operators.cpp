#include <cassert>

#include "operators.h"

namespace ad {

static void AddBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative();
    rhs->derivative() += val.derivative();
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

Var operator-(double a, const Var& v1) {
    Eigen::MatrixXd coeff(v1.value().rows(), v1.value().cols());
    coeff.setConstant(a);
    Var coeff_var = v1.graph()->CreateParam(coeff);
    return v1.graph()->CreateNode(
            a - v1.value().array(), coeff_var, v1, SubBackprop);
}

Var operator-(const Var& v1, double a) {
    Eigen::MatrixXd coeff(v1.value().rows(), v1.value().cols());
    coeff.setConstant(a);
    Var coeff_var = v1.graph()->CreateParam(coeff);
    return v1.graph()->CreateNode(
            v1.value().array() - a, v1, coeff_var, SubBackprop);
}

static void MulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() +=  val.derivative() * rhs->value().transpose();
    rhs->derivative() +=  lhs->value().transpose() * val.derivative();
}

static void CoeffMulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() +=  val.derivative() * rhs->value()(0, 0);
    rhs->derivatuve() += val.derivative().transpose() * lhs->value();
}

Var operator*(const Var& v1, const Var& v2) {
    if (v2.value().rows() == 1 && v2.value().cols() == 1) {
        return v2 * v1;
    }

    if (v1.value().rows() == 1 && v1.value().cols() == 1) {
        return v1.graph()->CreateNode(
                v2.value() * v1.value()(0, 0), v2, v1, CoeffMulBackprop);
    }

    return v1.graph()->CreateNode(v1.value() * v2.value(), v1, v2, MulBackprop);
}

Var operator*(double a, const Var& v1) {
    Eigen::MatrixXd coeff(1, 1);
    coeff << a;
    Var coeff_var = v1.graph()->CreateParam(coeff);
    return v1.graph()->CreateNode(v1.value() * a, v1, coeff_var, CoeffMulBackprop);
}

Var operator*(const Var& v1, double a) {
    Eigen::MatrixXd coeff(1, 1);
    coeff << a;
    Var coeff_var = v1.graph()->CreateParam(coeff);
    return v1.graph()->CreateNode(v1.value() * a, v1, coeff_var, CoeffMulBackprop);
}

static void ReluBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    const double* a = lhs->value().data();
    double* db = val.derivative().data();
    for (int i = 0; i < lhs->derivative().size(); ++i) {
        da[i] += a[i] > 0 ? db[i] : 0;
    }
}

Var Relu(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value().array().max(0), v1, no_operand, ReluBackprop);
}

static void SquareBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative() += 2 * val.derivative() * lhs->value();
}

Var Square(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value() * v1.value(), v1, no_operand, SquareBackprop);
}

static void EltSquareBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative() += 2 * val.derivative().cwiseProduct(lhs->value());
}

Var EltSquare(const Var& v1) {
    return v1.graph()->CreateNode(
            v1.value().cwiseProduct(v1.value()), v1, no_operand, EltSquareBackprop);
}

static void EltwiseMulBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative().cwiseProduct(rhs->value());
    rhs->derivative() += val.derivative().cwiseProduct(lhs->value());
}

Var operator^(const Var& v1, const Var& v2) {
    return v1.graph()->CreateNode(
            v1.value().cwiseProduct(v2.value()), v1, v2, EltwiseMulBackprop);
}

static void LogBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0; i < val.value().size(); ++i) {
        da[i] += dx[i] / a[i];
    }
}

Var Log(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = log(src_ptr[i]);
    }
    return x.graph()->CreateNode(res, x, no_operand, LogBackprop);
}

static void NLogBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0, size = val.value().size(); i < size; ++i) {
        da[i] += -dx[i] / (a[i]);
    }
}

Var NLog(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = -log(src_ptr[i]);
    }
    return x.graph()->CreateNode(res, x, no_operand, NLogBackprop);
}

Var CrossEntropy(const Var& h, const Var& y) {
    return Sum(y ^ NLog(h));
}

static Eigen::MatrixXd Softmax(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.data();
    double total = 0;
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = exp(src_ptr[i]);
        total += dst_ptr[i];
    }

    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] /= total;
    }
    return res;
}

static void SoftmaxLossBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative() += val.derivative().cwiseProduct(
            Softmax(lhs->value()) - rhs->value());
}

Var SoftmaxLoss(Var h, Var y) {
    auto res = Softmax(h.value());

    double* d = res.data();
    double* dy = y.value().data();
    for (size_t i = 0, len = res.size(); i < len; ++i) {
        if (dy[i] != 0.0) {
            d[i] = -dy[i] * log(d[i]);
        }
    }
    return h.graph()->CreateNode(res, h, y, SoftmaxLossBackprop);
}

static void ExpBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    const double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0; i < val.value().size(); ++i) {
        da[i] += dx[i] * exp(a[i]);
    }
}

Var Exp(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = exp(src_ptr[i]);
    }
    return x.graph()->CreateNode(res, x, no_operand, ExpBackprop);
}

static void SoftmaxBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    const double* dx = val.derivative().data();
    const double* a = val.value().data();
    size_t len = val.value().size();
    for (size_t i = 0; i < len; ++i) {
        double d = 0;
        for (size_t j = 0; j < len; ++j) {
            d += a[i] * (((i == j) ? 1 : 0) - a[j]) * dx[j];
        }
        da[i] += d;
    }
}

Var Softmax(const Var& x) {
    return x.graph()->CreateNode(
            Softmax(x.value()), x, no_operand, SoftmaxBackprop);
}

static void SigmoidBackprop(Var& val, Var* lhs, Var*) {
    const Eigen::MatrixXd& a = val.value();
    lhs->derivative() += val.derivative()
        .cwiseProduct((a.array() * (1.0 - a.array())).matrix());
}

Var Sigmoid(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = 1.0 / (1 + exp(-src_ptr[i]));
    }

    return x.graph()->CreateNode(res, x, no_operand, SigmoidBackprop);
}

static void SumBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative().array() += (double)val.derivative()(0, 0);
}

Var Sum(const Var& a) {
    Eigen::MatrixXd res(1, 1);
    res << a.value().sum();
    return a.graph()->CreateNode(res, a, no_operand, SumBackprop);
}

static void MeanBackprop(Var& val, Var* lhs, Var*) {
    lhs->derivative().array() +=
        (double)val.derivative()(0, 0) / val.value().size();
}

Var Mean(const Var& a) {
    Eigen::MatrixXd res(1, 1);
    res << a.value().sum() / a.value().size();
    return a.graph()->CreateNode(res, a, no_operand, MeanBackprop);
}

Var MSE(const Var& h, const Var& y) {
    return Mean(EltSquare(h - y));
}

Var SSE(const Var& h, const Var& y) {
    return Sum(EltSquare(h - y));
}

static void NthColBackprop(Var& val, Var* lhs, Var* rhs) {
    lhs->derivative().col(rhs->value()(0, 0)) += val.derivative();
}

Var NthCol(const Var& w, int n) {
    Eigen::MatrixXd n_mat(1, 1);
    n_mat << n;
    Var n_var = w.graph()->CreateParam(n_mat);
    return w.graph()->CreateNode(w.value().col(n), w, n_var, NthColBackprop);
}

static void TanhBackprop(Var& val, Var* lhs, Var*) {
    double* da = lhs->derivative().data();
    const double* dx = val.derivative().data();
    const double* a = lhs->value().data();
    for (int i = 0; i < val.value().size(); ++i) {
        double denom = std::cosh(2 * a[i]) + 1;
        denom = denom * denom;
        if (std::isfinite(denom)) {
            double num = std::cosh(a[i]);
            num = 4 * num * num;
            double res = num / denom;
            da[i] += dx[i] * res;
        }
    }
}

Var Tanh(const Var& x) {
    Eigen::MatrixXd res(x.value().rows(), x.value().cols());
    double* dst_ptr = res.data();
    const double* src_ptr = x.value().data();
    for (int i = 0; i < res.size(); ++i) {
        dst_ptr[i] = std::tanh(src_ptr[i]);
    }

    return x.graph()->CreateNode(res, x, no_operand, TanhBackprop);
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

    Eigen::MatrixXd cated(x.value().rows() + y.value().rows(), 1);
    cated << x.value(), y.value();
    return x.graph()->CreateNode(cated, x, y, ColAppendBackprop);
}

static void ColSplitBackprop(Var& val, Var* lhs, Var* params) {
    int from = params->value()(0, 0);
    int len = params->value()(1, 0);
    lhs->derivative().block(from, 0, len, 1) += val.derivative();
}

Var ColSplit(Var x, int from, int len) {
    if (x.value().cols() != 1) {
        throw std::runtime_error("cannot split not-a-column-vectors");
    }

    Eigen::MatrixXd params(2, 1);
    params << from, len;
    return x.graph()->CreateNode(
            x.value().block(from, 0, len, 1),
            x,
            x.graph()->CreateParam(params),
            ColSplitBackprop);
}

}
