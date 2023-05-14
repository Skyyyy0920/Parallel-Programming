//
// Created by Xuyao Wang on 5/5/2023.
//

#include "nn/ComputationalGraph/Operator/MulOperator.h"

std::vector<Tensor> MulOperator::backward(const Tensor& tensor,std::vector<Tensor> inputs){
    return {tensor*inputs[1], tensor*inputs[0]};
}

[[nodiscard]] MulOperator::OperatorType MulOperator::getType() const {
    return TYPE_MUL;
}