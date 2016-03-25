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

void WriteMatrix(const Matrix& gpumat, std::ostream& out) {
    char magic[] = "MATX";
    out.write((char*)&magic, 4);

    RWMatrix mat = gpumat.Fetch();
    int int_buf;
    int_buf = mat.rows();
    out.write((char*)&int_buf, sizeof (int));
    int_buf = mat.cols();
    out.write((char*)&int_buf, sizeof (int));
    out.write((char*)mat.data(), sizeof (double) * mat.size());
}

Matrix ReadMatrix(std::istream& out) {
    char magic[4];
    out.read((char*)&magic, 4);

    if (!std::strncpy(magic, "MATX", 4)) {
        throw std::invalid_argument("Magic number 'MATX' not present");
    }

    int rows, cols;
    out.read((char*)&rows, sizeof (int));
    out.read((char*)&cols, sizeof (int));

    RWMatrix mat(rows, cols);
    out.read((char*)mat.data(), sizeof (double) * mat.size());
    return Matrix(mat);
}

void WriteMatrixTxt(const Matrix& gpumat, std::ostream& out) {
    out << "MATX\n";

    RWMatrix mat = gpumat.Fetch();
    out << mat.rows() << " " << mat.cols() << "\n";
    const float* data = mat.data();
    for (size_t i = 0; i < mat.size(); ++i) {
        out << data[i] << " ";
    }
    out << "\n";
}

Matrix ReadMatrixTxt(std::istream& in) {
    std::string magic;
    in >> magic;

    if (magic != "MATX") {
        throw std::invalid_argument("Magic number 'MATX' not present");
    }

    int rows, cols;
    in >> rows >> cols;

    RWMatrix mat(rows, cols);
    float* data = mat.data();
    for (size_t i = 0; i < mat.size(); ++i) {
        in >> data[i];
    }

    return Matrix(mat);
}
} // utils
} // ad
