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

} // utils
} // ad
