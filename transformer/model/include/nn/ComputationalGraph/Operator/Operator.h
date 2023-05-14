//
// Created by Xuyao Wang on 5/4/2023.
//

#ifndef TRANSFORMER_OPERATOR_H
#define TRANSFORMER_OPERATOR_H

#include "nn/Tensor/Tensor.h"

class Operator {
protected:
    enum OperatorType {
        TYPE_ADD,
        TYPE_SUB,
        TYPE_MUL
    };
public:
    [[nodiscard]] virtual OperatorType getType() const = 0;
    virtual std::vector<Tensor> backward(const Tensor& tensor,std::vector<Tensor> inputs) = 0;

public:
    static OperatorType getAddType(){return TYPE_ADD;}
    static OperatorType getSubType(){return TYPE_SUB;}
    static OperatorType getMulType(){return TYPE_MUL;}


protected:
    // Prohibit external invocation of the constructor of the Operator class to
    // prevent derived classes from being created externally.
    Operator() = default;
};



#endif //TRANSFORMER_OPERATOR_H
