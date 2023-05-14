//
// Created by Xuyao Wang on 5/11/2023.
//

#include "nn/ComputationalGraph/Operator/SubOperator.h"

std::vector<Tensor> SubOperator::backward(const Tensor& tensor,std::vector<Tensor> inputs){
    return {tensor, -tensor};
}

SubOperator::OperatorType SubOperator::getType() const {
    return TYPE_SUB;
}