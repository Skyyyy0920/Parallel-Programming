#include <iostream>
#include "Transformer/Transformer.h"
#include "model/include/nn/Tensor/Tensor.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor result = t1 - t2;
    result.print();
    return 0;
}
