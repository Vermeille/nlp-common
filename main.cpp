#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

#include <ad/ad.h>

int main() {
    using namespace ad;
    Var a(1, 2, VarType::Param);
    a.val() << 3, 1;
    Var b(1, 1, VarType::Param);
    b.val() << 6;
    for (int i = 1; i < 100; ++i) {
        Var j(1, 1);
        j.val() << 0;
        for (int k = 0; k < 10; ++k) {
            Var x(2, 1);
            x.val() << (rand() % 30 - 15) / 15., (rand() % 30 - 15) / 15.;
            Var y(1, 1);
            y.val() << x.val()(0, 0) * 8 + x.val()(1, 0) * 3 + 5;

            Var h = a * x + b; // model
            j = MSE(h, y) + j; // hypothesis
        }
        std::cout << "COST =\n" << j.val() << "\n";
        j.Backprop();
        a.val() -= 0.1 * a.derivative() / 10;
        b.val() -= 0.1 * b.derivative() / 10;
        j.ClearGrad();
    }
    return 0;
}

