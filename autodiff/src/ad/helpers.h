#pragma once

#include <iostream>

#include <Eigen/Dense>

namespace ad {
namespace utils {

double RandomRange(float from, float to);
void RandomInit(Eigen::MatrixXd& mat, float from, float to);
void RandomExpandMatrix(Eigen::MatrixXd& mat, int rows, int cols,
        float from, float to);
Eigen::MatrixXd OneHotColumnVector(int index, int rows);
int OneHotVectorDecode(const Eigen::MatrixXd& mat);

void WriteMatrix(const Eigen::MatrixXd& mat, std::ostream& out);
Eigen::MatrixXd ReadMatrix(std::istream& out);

void WriteMatrixTxt(const Eigen::MatrixXd& mat, std::ostream& out);
Eigen::MatrixXd ReadMatrixTxt(std::istream& out);

} // utils
} // ad
