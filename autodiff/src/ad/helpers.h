#pragma once

#include <iostream>
#include "matrix.h"

namespace ad {
namespace utils {

double RandomRange(float from, float to);
void RandomExpandMatrix(Matrix& mat, int rows, int cols, float from, float to);
Matrix OneHotColumnVector(int index, int rows);
int OneHotVectorDecode(const Matrix& mat);

void WriteMatrix(const Matrix& mat, std::ostream& out);
Matrix ReadMatrix(std::istream& out);

void WriteMatrixTxt(const Matrix& mat, std::ostream& out);
Matrix ReadMatrixTxt(std::istream& out);

} // utils
} // ad
