//
// Created by Xuyao Wang on 5/5/2023.
//

#include "nn/ComputationalGraph/Var/AutoDifferentialVar.h"
void AutoDifferentialVar::backward(const Tensor& gradient){
    double learning_rate=0.001;

    if (dataInfo.op== nullptr){
        dataInfo.data=dataInfo.data-gradient*learning_rate;
        return;
    }
    std::vector<Tensor>grads=dataInfo.op->backward(gradient,
                                                   std::vector<Tensor>{dataInfo.predecessorNodes[0]->getData(),
                                                                       dataInfo.predecessorNodes[1]->getData()});

    for (int i = 0; i < dataInfo.predecessorNodes.size(); ++i) {
        dynamic_cast<GradientInterface*>(dataInfo.predecessorNodes[i])->backward(grads[i]);
    }
}

void AutoDifferentialVar::backward(){
    this->backward(Tensor(this->getData().getShape(),1.0));
}

bool AutoDifferentialVar::operator==(const Var& other) const{
    auto var=dynamic_cast<const AutoDifferentialVar*>(&other);
    return (var!= nullptr)&&(Var::operator==(other));
}

void AutoDifferentialVar::print(){
    std::cout<<"auto-differentiable Var\n";
    std::cout<<"=================================\n";
    std::cout<<"data:\n";
    dataInfo.data.print();
    if (dataInfo.op)
    std::cout<<"Operator:\n"<<dataInfo.op->getType()<<std::endl;
    std::cout<<"=================================\n\n";
}

bool AutoDifferentialVar::isAutoDifferentiable() {
    return true;
}
