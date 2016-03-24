#pragma once

#include "matrix.h"

namespace ad {
namespace cuda {

void SetIdentity(float* array, size_t rows, size_t cols);

}
}

namespace ad {

struct MatrixInitialization {
    virtual void Init(Matrix& mat) const = 0;
};

class Constant : public MatrixInitialization {
    private:
        double cst_;
    public:
        virtual void Init(Matrix& mat) const {
            mat.SetConstant(cst_);
        }

        Constant(double cst) : cst_(cst) {}
};

class Gaussian : public MatrixInitialization {
    private:
        float mu_;
        float sigma_;
    public:
        virtual void Init(Matrix& mat) const;
        Gaussian(double mu, double sigma)
            : mu_(mu),
            sigma_(sigma) {
        }
};

struct Uniform : public MatrixInitialization {
    private:
        float from_;
        float to_;
    public:
        virtual void Init(Matrix& mat) const;
        Uniform(double from, double to)
            : from_(from),
            to_(to) {
        }
};

struct Xavier : public MatrixInitialization {
    virtual void Init(Matrix& mat) const {
        Uniform uni(0, 2.0 / sqrt(mat.rows() + mat.cols()));
        uni.Init(mat);
    }
};

struct Identity : public MatrixInitialization {
    virtual void Init(Matrix& mat) const {
        mat.SetIdentity();
    }
};

} // ad
