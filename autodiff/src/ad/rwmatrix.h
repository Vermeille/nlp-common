#pragma once

namespace ad {

/*
 * The CPU counterpart of a Matrix (on GPU). Used as an editable proxy before
 * uploading the matrices to GPU or after having fetched them from the CPU.
 */
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

} // ad
