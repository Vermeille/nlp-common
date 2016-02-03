#include "helpers.h"

#include <cstdlib>
#include <cassert>

namespace ad {
namespace utils {

double RandomRange(float from, float to) {
    double distance = to - from;
    return ((double)rand() / ((double)RAND_MAX + 1) * distance) + from;
}

void RandomInit(Eigen::MatrixXd& mat, float from, float to) {
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            mat(i, j) = RandomRange(from, to);
        }
    }
}

void RandomExpandMatrix(Eigen::MatrixXd& mat, int rows, int cols,
        float from, float to) {
    int old_rows = mat.rows();
    int old_cols = mat.cols();
    mat.conservativeResize(rows, cols);

    if (cols > old_cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = old_cols; j < cols; ++j) {
                mat(i, j) = RandomRange(from, to);
            }
        }
    }

    if (rows > old_rows) {
        for (int i = old_rows; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat(i, j) = RandomRange(from, to);
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
        throw std::invalid_argument("Magix number 'MATX' not present");
    }

    int rows, cols;
    out.read((char*)&rows, sizeof (int));
    out.read((char*)&cols, sizeof (int));

    Eigen::MatrixXd mat(rows, cols);
    out.read((char*)mat.data(), sizeof (double) * mat.size());
    return mat;
}
} // utils
} // ad
