#include "matrix.h"
#include "kernels/kernels.h"

namespace ad {

thread_local GPUChunksPool g_gpu_pool;

void Matrix::Clip(double clip) {
    float norm;
    cublasSnrm2(::cuda::g_cuhandle.get(), size(), data().Get(), 1, &norm);
    if (norm > clip) {
        auto x = cuda::Array(data());
        cuda::RunKernel(cuda::Seq(
                x = (x / cuda::Value(norm)) * cuda::Value(clip)),
            size());
    }
}

void operator+=(Matrix& m, const Matrix& n) {
    // FIXME implement broadcasting
    assert(m.rows() == n.rows() && m.cols() == n.cols());
    auto cum = cuda::Array(m.data());
    auto cun = cuda::Array(n.data());
    cuda::RunKernel(cum += cun, m.size());
}

Matrix& operator+=(Matrix& m, float a) {
    auto cum = cuda::Array(m.data());
    auto cua = cuda::Value(a);
    cuda::RunKernel(cum += cua, m.size());
    return m;
}

void operator-=(Matrix& m, const Matrix& n) {
    assert(m.rows() == n.rows() && m.cols() == n.cols());
    auto cum = cuda::Array(m.data());
    auto cun = cuda::Array(n.data());
    cuda::RunKernel(cum -= cun, m.size());
}

Matrix operator+(const Matrix& m, const Matrix& n) {
    assert(m.rows() == n.rows() && m.cols() == n.cols());
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cun = cuda::Array(n.data());
    cuda::RunKernel(cuda::Eq(cur, cum + cun), res.size());
    return res;
}

Matrix operator+(const Matrix& m, float a) {
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cua = cuda::Value(a);
    cuda::RunKernel(cuda::Eq(cur, cum + cua), res.size());
    return res;
}

Matrix operator-(const Matrix& m, const Matrix& n) {
    assert(m.rows() == n.rows() && m.cols() == n.cols());
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cun = cuda::Array(n.data());
    cuda::RunKernel(cuda::Eq(cur, cum - cun), res.size());
    return res;
}

Matrix operator-(const Matrix& m, float a) {
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cua = cuda::Value(a);
    cuda::RunKernel(cuda::Eq(cur, cum - cua), res.size());
    return res;
}

Matrix operator-(float a, const Matrix& m) {
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cua = cuda::Value(a);
    cuda::RunKernel(cuda::Eq(cur, cua - cum), res.size());
    return res;
}

Matrix operator^(const Matrix& m, const Matrix& n) {
    assert(m.rows() == n.rows() && m.cols() == n.cols());
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cun = cuda::Array(n.data());
    cuda::RunKernel(cuda::Eq(cur, cum * cun), res.size());
    return res;
}

Matrix operator*(float a, const Matrix& m) {
    Matrix res(m.rows(), m.cols());
    auto cur = cuda::Array(res.data());
    auto cum = cuda::Array(m.data());
    auto cua = cuda::Value(a);
    cuda::RunKernel(cuda::Eq(cur, cum * cua), res.size());
    return res;
}

} // ad
