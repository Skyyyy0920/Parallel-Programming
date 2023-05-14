//
// Created by Xuyao Wang on 5/5/2023.
//

#include "nn/ComputationalGraph/Var/Var.h"

bool Var::operator==(const Var& other) const{
    return  (this->getData()==other.getData())&&
            (this->getOp()==other.getOp());
};

bool Var::operator!=(const Var& other) const{
    return !operator==(other);
};


Var::DataInfo Var::Mul(const Var& other) const{
    Tensor result_data = dataInfo.data*other.getData();
    Operator*newOp=OperatorFactory::getInstance().getMulOperator();
    return DataInfo(result_data, newOp,std::vector<Var*>({const_cast<Var*>(this),const_cast<Var*>(&other)}));
}

Var::DataInfo Var::Add(const Var& other) const{
    Tensor result_data = dataInfo.data+other.getData();
    Operator*newOp=OperatorFactory::getInstance().getAddOperator();
    return DataInfo(result_data, newOp,std::vector<Var*>({const_cast<Var*>(this),const_cast<Var*>(&other)}));
};

Var::DataInfo Var::Sub(const Var &other) const {
    Tensor result_data = dataInfo.data-other.getData();
    Operator*newOp=OperatorFactory::getInstance().getSubOperator();
    return DataInfo(result_data, newOp,std::vector<Var*>({const_cast<Var*>(this),const_cast<Var*>(&other)}));
};

[[nodiscard]] const Tensor& Var::getData() const {
    return dataInfo.data;
}

void Var::setData(const Tensor &tensor) {
    Var::dataInfo.data = tensor;
}

[[nodiscard]] Operator *Var::getOp() const {
    return dataInfo.op;
}
const std::vector<Var*>& Var::getPredecessorNodes() const {
    return dataInfo.predecessorNodes;
}

void Var::setPredecessorNodes(const std::vector<Var*>& predecessorNodes) {
    this->dataInfo.predecessorNodes = predecessorNodes;
}