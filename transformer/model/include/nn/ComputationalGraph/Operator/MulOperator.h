//
// Created by Xuyao Wang on 5/4/2023.
//

#ifndef TRANSFORMER_MULOPERATOR_H
#define TRANSFORMER_MULOPERATOR_H

#include "Operator.h"

class MulOperator:public Operator{
public:
    std::vector<Tensor> backward(const Tensor& grad_output,std::vector<Tensor> inputs) override;
    [[nodiscard]] OperatorType getType() const override;
private:
    MulOperator()=default;
    friend class OperatorFactory;
};

#endif //TRANSFORMER_MULOPERATOR_H
