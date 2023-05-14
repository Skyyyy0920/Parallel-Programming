//
// Created by Xuyao Wang on 5/5/2023.
//

#include "nn/ComputationalGraph/Var/NonAutoDifferentialVar.h"

bool NonAutoDifferentialVar::operator==(const Var& other) const{
    auto var=dynamic_cast<const NonAutoDifferentialVar*>(&other);
    return (var!= nullptr)&&(this->getData()==other.getData());
}

bool NonAutoDifferentialVar::operator!=(const Var& other) const{
    return !operator!=(other);
}

void NonAutoDifferentialVar::backward(const Tensor& gradient){}

void NonAutoDifferentialVar::backward() {}

void NonAutoDifferentialVar::print(){
std::cout<<"non-auto-differentiable Var:\n";
std::cout<<"=================================\n";
std::cout<<"data:\n";
dataInfo.data.print();
std::cout<<"=================================\n\n";
}

bool NonAutoDifferentialVar::isAutoDifferentiable() {
    return false;
}
