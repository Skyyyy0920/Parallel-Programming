//
// Created by Xuyao Wang on 5/11/2023.
//

#ifndef TRANSFORMER_SUBOPERATOR_H
#define TRANSFORMER_SUBOPERATOR_H

#include "Operator.h"

class SubOperator:public Operator{
public:
    std::vector<Tensor> backward(const Tensor& tensor,std::vector<Tensor> inputs) override;
    [[nodiscard]] OperatorType getType() const override;
private:
    friend class OperatorFactory;
    SubOperator()=default;
};
#endif //TRANSFORMER_SUBOPERATOR_H
