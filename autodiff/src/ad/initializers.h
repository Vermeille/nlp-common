#pragma once

#include <Eigen/Dense>

namespace ad {

struct MatrixInitialization {
    virtual void Init(Eigen::MatrixXd& mat) const = 0;
};

class Constant : public MatrixInitialization {
    private:
        double cst_;
    public:
        virtual void Init(Eigen::MatrixXd& mat) const {
            mat.setConstant(cst_);
        }

        Constant(double cst) : cst_(cst) {}
};

class Gaussian : public MatrixInitialization {
    private:
        double mu_;
        double sigma_;
    public:
        virtual void Init(Eigen::MatrixXd& mat) const;
        Gaussian(double mu, double sigma)
            : mu_(mu),
            sigma_(sigma) {
        }
};

struct Uniform : public MatrixInitialization {
    private:
        double from_;
        double to_;
    public:
        virtual void Init(Eigen::MatrixXd& mat) const;
        Uniform(double from, double to)
            : from_(from),
            to_(to) {
        }
};

struct Xavier : public MatrixInitialization {
    virtual void Init(Eigen::MatrixXd& mat) const {
        Uniform uni(0, 2.0 / (mat.rows() + mat.cols()));
        uni.Init(mat);
    }
};

} // ad
