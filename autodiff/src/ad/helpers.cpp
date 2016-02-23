#include "helpers.h"

#include <cstdlib>
#include <cassert>
#include <random>

namespace ad {
namespace utils {

double RandomRange(float from, float to) {
    double distance = to - from;
    return ((double)rand() / ((double)RAND_MAX + 1) * distance) + from;
}

void RandomExpandMatrix(Eigen::MatrixXd& mat, int rows, int cols,
        float from, float to) {
    int old_rows = mat.rows();
    int old_cols = mat.cols();
    mat.conservativeResize(rows, cols);
    std::default_random_engine generator;
    std::normal_distribution<double> gaussian(0, 2.0 / (mat.rows() + mat.cols()));

    if (cols > old_cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = old_cols; j < cols; ++j) {
                mat(i, j) = gaussian(generator);
            }
        }
    }

    if (rows > old_rows) {
        for (int i = old_rows; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat(i, j) = gaussian(generator);
            }
        }
    }
}

Eigen::MatrixXd OneHotColumnVector(int index, int rows) {
    Eigen::MatrixXd mat(rows, 1);
    mat.setZero();
    mat(index, 0) = 1;
    return mat;
}

int OneHotVectorDecode(const Eigen::MatrixXd& mat) {
    assert(mat.rows() == 1 || mat.cols() == 1);

    Eigen::MatrixXd::Index max_row, max_col;
    mat.maxCoeff(&max_row, &max_col);
    if (mat.rows() == 1) {
        return max_col;
    } else {
        return max_row;
    }
}

void WriteMatrix(const Eigen::MatrixXd& mat, std::ostream& out) {
    uint32_t magic = 'MATX';
    out.write((char*)&magic, 4);

    int int_buf;
    int_buf = mat.rows();
    out.write((char*)&int_buf, sizeof (int));
    int_buf = mat.cols();
    out.write((char*)&int_buf, sizeof (int));
    out.write((char*)mat.data(), sizeof (double) * mat.size());
}

Eigen::MatrixXd ReadMatrix(std::istream& out) {
    uint32_t magic;
    out.read((char*)&magic, 4);

    if (magic != 'MATX') {
        throw std::invalid_argument("Magic number 'MATX' not present");
    }

    int rows, cols;
    out.read((char*)&rows, sizeof (int));
    out.read((char*)&cols, sizeof (int));

    Eigen::MatrixXd mat(rows, cols);
    out.read((char*)mat.data(), sizeof (double) * mat.size());
    return mat;
}

void WriteMatrixTxt(const Eigen::MatrixXd& mat, std::ostream& out) {
    out << "MATX\n";

    out << mat.rows() << " " << mat.cols() << "\n";
    const double* data = mat.data();
    for (size_t i = 0; i < mat.size(); ++i) {
        out << data[i] << " ";
    }
    out << "\n";
}

Eigen::MatrixXd ReadMatrixTxt(std::istream& in) {
    std::string magic;
    in >> magic;

    if (magic != "MATX") {
        throw std::invalid_argument("Magic number 'MATX' not present");
    }

    int rows, cols;
    in >> rows >> cols;

    Eigen::MatrixXd mat(rows, cols);
    double* data = mat.data();
    for (size_t i = 0; i < mat.size(); ++i) {
        in >> data[i];
    }

    return mat;
}
} // utils
} // ad
