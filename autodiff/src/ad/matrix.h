#pragma once

#include <cassert>
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "cuda/helpers.h"

namespace ad {
namespace cuda {

void SetIdentity(float* array, size_t rows, size_t cols);

}
}

namespace ad {

class RWMatrix {
    size_t rows_;
    size_t cols_;
    std::unique_ptr<float[]> data_;

    public:
        RWMatrix(size_t r, size_t c)
            : rows_(r),
            cols_(c),
            data_(new float[r * c]) {
        }

        size_t size() const { return rows_ * cols_; }
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        float* data() { return data_.get(); }
        const float* data() const { return data_.get(); }

        void SetConstant(float f) {
            std::fill(data_.get(), data_.get() + size(), f);
        }

        void SetOnes() {
            SetConstant(1);
        }

        void SetZero() {
            SetConstant(0);
        }

        float& operator()(int i, int j) {
            return data_[i + j * rows_];
        }

        float operator()(int i, int j) const {
            return data_[i + j * rows_];
        }
};

inline
std::ostream& operator<<(std::ostream& strm, const RWMatrix& m) {
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            strm << m(i, j) << " ";
        }
        strm << std::endl;
    }
    return strm;
}

// 15% speedup
class GPUChunksPool {
    public:
        template <class T>
        using cuptr = ::cuda::Ptr<T>;

    private:
        std::unordered_map<int, std::vector<cuptr<float>>> frees_;
        std::unordered_map<cuptr<float>, size_t> allocated_;

    public:
        ::cuda::Ptr<float> Alloc(size_t sz) {
            auto& frees = frees_[sz];
            cuptr<float> res;
            if (frees.empty()) {
                res = ::cuda::helpers::cunew<float>(sz);
            } else {
                res = frees.back();
                frees.pop_back();
            }
            allocated_[res] = sz;
            return res;
        }

        void Free(cuptr<float> ptr) {
            auto sz_iter = allocated_.find(ptr);
            if (sz_iter == allocated_.end()) {
                throw std::runtime_error("try to free a ptr not allocated");
            }
            size_t sz = sz_iter->second;
            allocated_.erase(sz_iter);
            frees_[sz].push_back(ptr);
        }
};

extern thread_local GPUChunksPool g_gpu_pool;

struct GPUPoolDeleter {
    typedef ::cuda::Ptr<float> pointer;
    void operator()(::cuda::Ptr<float> ptr) {
        g_gpu_pool.Free(ptr);
    }
};

class Matrix {
    template <class T>
    using cuptr = ::cuda::Ptr<T>;

    size_t rows_;
    size_t cols_;
    std::unique_ptr<cuptr<float>[], GPUPoolDeleter> data_;

    public:
        Matrix(size_t r, size_t c)
            : rows_(r),
            cols_(c),
            data_(g_gpu_pool.Alloc(r * c)) {
        }

        explicit Matrix(const RWMatrix& m)
            : rows_(m.rows()),
            cols_(m.cols()),
            data_(g_gpu_pool.Alloc(rows_ * cols_)) {
                ::cuda::helpers::CPUToGPU(data(), m.data(), size());
        }

        size_t size() const { return rows_ * cols_; }
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        cuptr<float> data() { return data_.get(); }
        const cuptr<float> data() const { return data_.get(); }

        Matrix(Matrix&& m)
            : rows_(m.rows_),
            cols_(m.cols_),
            data_(std::move(m.data_)) {
        }

        Matrix& operator=(Matrix&& m) {
            rows_ = m.rows_;
            cols_ = m.cols_;
            data_ = std::move(m.data_);
            m.data_.reset(g_gpu_pool.Alloc(m.rows_ * m.cols_));
            return *this;
        }

        RWMatrix Fetch() const {
            RWMatrix res(rows_, cols_);
            ::cuda::helpers::GPUToCPU(res.data(), data(), size());
            return res;
        }

        void SetConstant(float f) {
            ::cuda::helpers::fill(data_.get(), size(), f);
        }

        void SetOnes() {
            SetConstant(1);
        }

        void SetZero() {
            cudaMemset(data_.get().Get(), 0, size() * sizeof (float));
        }

        void SetIdentity() {
            cuda::SetIdentity(data_.get().Get(), rows_, cols_);
        }

        float CudaRead(size_t i, size_t j) const {
            float res;
            ::cuda::helpers::GPUToCPU(&res, data_.get().At(i + j * rows_), 1);
            return res;
        }

        void CudaWrite(size_t i, size_t j, float v) {
            ::cuda::helpers::CPUToGPU(data_.get().At(i + j * rows_), &v, 1);
        }

        float sum() const {
            float res;
            cublasSasum(
                    ::cuda::g_cuhandle.get(),
                    size(), data_.get().Get(), 1, &res);
            return res;
        }

        Matrix block(size_t a, size_t b, size_t m, size_t n) {
            Matrix res(m, n);
            cuptr<float> dst = res.data();
            for (size_t i = 0; i < n; ++i) {
                ::cuda::helpers::GPUToGPU(
                        dst,
                        data_.get().At((b + i) * rows_ + a), m);
                dst = cuptr<float>(dst.Get() + m);
            }
            return res;
        }

        void Clip(double clip);
};

void operator+=(Matrix& m, const Matrix& n);
Matrix& operator+=(Matrix& m, float a);
void operator-=(Matrix& m, const Matrix& n);
Matrix operator+(const Matrix& m, const Matrix& n);
Matrix operator+(const Matrix& m, float a);

inline
Matrix operator+(float a, const Matrix& m) {
    return m + a;
}

Matrix operator-(const Matrix& m, const Matrix& n);
Matrix operator-(const Matrix& m, float a);
Matrix operator-(float a, const Matrix& m);
Matrix operator^(const Matrix& m, const Matrix& n);

inline
Matrix operator*(const Matrix& a, const Matrix& b) {
    assert(a.cols() == b.rows());
    Matrix res(a.rows(), b.cols());
    size_t m = res.rows();
    size_t n = res.cols();
    size_t k = a.cols();
    float one = 1.f;
    float zero = 0.f;
    cublasSgemm(
            ::cuda::g_cuhandle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &one,
            a.data().Get(), a.rows(),
            b.data().Get(), b.rows(),
            &zero,
            res.data().Get(), m);
    return res;
}

Matrix operator*(float a, const Matrix& b);

inline
Matrix operator*(const Matrix& b, float a) {
    return a * b;
}

inline
Matrix MulNN(const Matrix& a, const Matrix& b) {
    return a * b;
}

inline
Matrix MulNT(const Matrix& a, const Matrix& b) {
    assert(a.cols() == b.cols());
    Matrix res(a.rows(), b.rows());
    size_t m = res.rows();
    size_t n = res.cols();
    size_t k = a.cols();
    float one = 1.f;
    float zero = 0.f;
    cublasSgemm(
            ::cuda::g_cuhandle.get(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            m, n, k,
            &one,
            a.data().Get(), a.rows(),
            b.data().Get(), b.rows(),
            &zero, res.data().Get(), m);
    return res;
}

inline
Matrix MulTN(const Matrix& a, const Matrix& b) {
    assert(a.rows() == b.rows());
    Matrix res(a.cols(), b.cols());
    size_t m = res.rows();
    size_t n = res.cols();
    size_t k = a.rows();
    float one = 1.f;
    float zero = 0.f;
    cublasSgemm(
            ::cuda::g_cuhandle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            &one,
            a.data().Get(), a.rows(),
            b.data().Get(), b.rows(),
            &zero,
            res.data().Get(), m);
    return res;
}

inline
Matrix MulTT(const Matrix& a, const Matrix& b) {
    assert(a.rows() == b.cols());
    Matrix res(a.cols(), b.rows());
    size_t m = res.rows();
    size_t n = res.cols();
    size_t k = a.rows();
    float one = 1.f;
    float zero = 0.f;
    cublasSgemm(
            ::cuda::g_cuhandle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            m, n, k,
            &one, a.data().Get(), m, b.data().Get(), k,
            &zero, res.data().Get(), m);
    return res;
}

} // ad
