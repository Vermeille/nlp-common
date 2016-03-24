#include "helpers.h"

#include <cstdlib>
#include <cassert>
#include <random>

#include "cuda/helpers.h"

namespace ad {
namespace utils {

double RandomRange(float from, float to) {
    double distance = to - from;
    return ((double)rand() / ((double)RAND_MAX + 1) * distance) + from;
}

Matrix OneHotColumnVector(int index, int rows) {
    RWMatrix mat(rows, 1);
    mat.SetZero();
    mat(index, 0) = 1;
    return Matrix(mat);
}

int OneHotVectorDecode(const Matrix& mat) {
    assert(mat.rows() == 1 || mat.cols() == 1);

    int idx;
    cublasIsamax(
            ::cuda::g_cuhandle.get(), mat.size(), mat.data().Get(), 1, &idx);
    return idx - 1;
}

void WriteMatrix(const Matrix& mat, std::ostream& out) {
#if 0
    uint32_t magic = 'MATX';
    out.write((char*)&magic, 4);

    int int_buf;
    int_buf = mat.rows();
    out.write((char*)&int_buf, sizeof (int));
    int_buf = mat.cols();
    out.write((char*)&int_buf, sizeof (int));
    out.write((char*)mat.data(), sizeof (double) * mat.size());
#endif
}

Matrix ReadMatrix(std::istream& out) {
#if 0
    uint32_t magic;
    out.read((char*)&magic, 4);

    if (magic != 'MATX') {
        throw std::invalid_argument("Magic number 'MATX' not present");
    }

    int rows, cols;
    out.read((char*)&rows, sizeof (int));
    out.read((char*)&cols, sizeof (int));

    Matrix mat(rows, cols);
    out.read((char*)mat.data(), sizeof (double) * mat.size());
    return mat;
#endif
    return Matrix(1, 1);
}

void WriteMatrixTxt(const Matrix& mat, std::ostream& out) {
#if 0
    out << "MATX\n";

    out << mat.rows() << " " << mat.cols() << "\n";
    const float* data = mat.data();
    for (size_t i = 0; i < mat.size(); ++i) {
        out << data[i] << " ";
    }
    out << "\n";
#endif
}

Matrix ReadMatrixTxt(std::istream& in) {
#if 0
    std::string magic;
    in >> magic;

    if (magic != "MATX") {
        throw std::invalid_argument("Magic number 'MATX' not present");
    }

    int rows, cols;
    in >> rows >> cols;

    Matrix mat(rows, cols);
    double* data = mat.data();
    for (size_t i = 0; i < mat.size(); ++i) {
        in >> data[i];
    }

    return mat;
#endif
    return Matrix(1, 1);
}
} // utils
} // ad
