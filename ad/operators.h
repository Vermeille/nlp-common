#pragma once

#include <memory>

#include "var.h"

namespace ad {

Var operator+(const Var& v1, const Var& v2);
Var operator-(const Var& v1, const Var& v2);
Var operator*(const Var& v1, const Var& v2);
Var operator^(const Var& a, const Var& b);
Var Relu(const Var& v1);
Var Square(const Var& v1);
Var Log(const Var& x);
Var NLog(const Var& x);
Var MSE(const Var& h, const Var& y);
Var CrossEntropy(const Var& h, const Var& y);

}
