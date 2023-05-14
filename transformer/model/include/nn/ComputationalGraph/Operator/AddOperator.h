//
// Created by Xuyao Wang on 5/4/2023.
//

#ifndef TRANSFORMER_ADDOPERATOR_H
#define TRANSFORMER_ADDOPERATOR_H

#include "Operator.h"

class AddOperator:public Operator{
public:
    std::vector<Tensor> backward(const Tensor& tensor,std::vector<Tensor> inputs) override;
    [[nodiscard]] OperatorType getType() const override;
private:
    friend class OperatorFactory;
    AddOperator()=default;
};

#endif //TRANSFORMER_ADDOPERATOR_H
