#pragma once

#include <cstdio>

#include "../cuda/helpers.h"

#ifdef __CUDACC__
# define CUDA_CALLABLE __host__ __device__
#else
# define CUDA_CALLABLE
#endif

namespace ad {
namespace cuda {

template <class T>
class K {
    T x_;

    public:
        template <class... Ts>
        K(Ts&&... xs) : x_(xs...) {}

        CUDA_CALLABLE
        auto operator()(size_t i) -> decltype (x_(i)) { return x_(i); }
        template <class U>
        auto operator=(const U& x) -> decltype (Eq(*this, x)) { return Eq(*this, x); }
};

#define BINOP(Name, Binop) \
template <class T, class U> \
class Name##Op { \
    T lhs_; \
    U rhs_; \
\
    public: \
        Name##Op(const T& lhs, const U& rhs) : lhs_(lhs), rhs_(rhs) {} \
\
        CUDA_CALLABLE \
        inline float operator()(size_t i) {\
            return lhs_(i) Binop rhs_(i); \
        }\
};\
\
template <class T, class U>\
K<Name##Op<T, U>> Name(const T& x, const U& y) { return K<Name##Op<T, U>>(x, y); } \

BINOP(Add, +)
BINOP(Sub, -)
BINOP(Mul, *)
BINOP(Div, /)
BINOP(Eq, =)
BINOP(AddEq, +=)
BINOP(SubEq, -=)
BINOP(Greater, >)

#undef BINOP

#define BINOP(Name, Binop) \
template <class T, class U>\
K<Name##Op<T, U>> operator Binop(const T& x, const U& y) { return K<Name##Op<T, U>>(x, y); } \

BINOP(Add, +)
BINOP(Sub, -)
BINOP(Mul, *)
BINOP(Div, /)
BINOP(AddEq, +=)
BINOP(SubEq, -=)
BINOP(Greater, >)

#undef BINOP

template <class T>
class KArray {
    T* arr_;

    public:
        KArray(::cuda::Ptr<T> a) : arr_(a.Get()) {}

        CUDA_CALLABLE
        inline T& operator()(size_t i) { return arr_[i]; }
};

template <class T>
K<KArray<T>> Array(::cuda::Ptr<T> xs) { return K<KArray<T>>(xs); }

class KValue {
    float val_;

    public:
        KValue(float a) : val_(a) {}

        CUDA_CALLABLE
        inline float& operator()(size_t) { return val_; }
};

inline
K<KValue> Value(float f) { return K<KValue>(f); }

template <class F, class T>
class KFun {
    T arg_;

    public:
        KFun(const T& arg) : arg_(arg) {}

        CUDA_CALLABLE
        inline float operator()(size_t i) {
            F f;
            return f(arg_(i));
        }
};

template <class F, class T>
K<KFun<F, T>> Fun(const T& a) { return K<KFun<F, T>>(a); }

#define DEFUN(Name) \
struct Name##Op { \
    __device__ float operator()(float x) { return Name(x); } \
}; \
\
template <class T>\
K<KFun<Name##Op, T>> Name(const T& a) { return K<KFun<Name##Op, T>>(a); } \

DEFUN(sqrt)
DEFUN(exp)
DEFUN(tanh)

#undef DEFUN

struct logOp {
    __device__ float operator()(float x) { return log(max(x, 1e-10f)); }
};

template <class T>
K<KFun<logOp, T>> log(const T& a) { return K<KFun<logOp, T>>(a); } \

template <class A1, class A2>
class KSeq {
    A1 a1_;
    A2 a2_;

    public:
        KSeq(const A1& a1, const A2& a2) : a1_(a1), a2_(a2) {}

        CUDA_CALLABLE
        inline void operator()(size_t i) {
            a1_(i);
            a2_(i);
        }
};

template <class A1>
K<A1> Seq(const A1& a1) {
    return a1;
}

template <class A1, class... As>
auto Seq(const A1& a1, As&&... as) -> K<KSeq<A1, decltype (Seq(as...))>> {
    return K<KSeq<A1, decltype (Seq(as...))>>(a1, Seq(as...));
}

template <class Op>
__global__
void GenericKernel(Op op, size_t sz) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < sz; i += blockDim.x * gridDim.x) {
        op(i);
    }
}

template <class Op>
void RunKernel(const Op& op, size_t sz) {
    const int nbthreads = 256;
    GenericKernel<Op><<<(sz + nbthreads - 1) / nbthreads, nbthreads>>>(op, sz);
}

void SetIdentity(float* array, size_t rows, size_t cols);

void Relu(::cuda::Ptr<float> res, const ::cuda::Ptr<float> arr1, size_t sz);
void ReluGrad(::cuda::Ptr<float> res, const ::cuda::Ptr<float> arr1, size_t sz);

} // cuda
} // ad
