//
// Created by Xuyao Wang on 5/5/2023.
//
#include "nn/ComputationalGraph/Operator/AddOperator.h"

std::vector<Tensor> AddOperator::backward(const Tensor& tensor,std::vector<Tensor> inputs){
    return {tensor, tensor};
}

AddOperator::OperatorType AddOperator::getType() const {
    return TYPE_ADD;
}