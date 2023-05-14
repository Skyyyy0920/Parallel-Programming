//
// Created by Xuyao Wang on 5/5/2023.
//

#ifndef TRANSFORMER_GRADIENTINTERFACE_H
#define TRANSFORMER_GRADIENTINTERFACE_H

#include "nn/Tensor/Tensor.h"

class GradientInterface{
public:
    GradientInterface()=default;
    explicit GradientInterface(double learnable):learnable(learnable){}
    virtual void backward(const Tensor& gradient)=0;
    virtual void backward()=0;
    virtual bool isAutoDifferentiable()=0;


    double getLearnable() const {
        return learnable;
    }

protected:
    double learnable;
};

#endif //TRANSFORMER_GRADIENTINTERFACE_H
