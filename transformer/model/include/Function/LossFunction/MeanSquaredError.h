//
// Created by Xuyao Wang on 5/11/2023.
//

#ifndef TRANSFORMER_MEANSQUAREDERROR_H
#define TRANSFORMER_MEANSQUAREDERROR_H

#include "LossFunction.h"

class MeanSquaredError:public LossFunction{
public:
    LossFunctionType getType() const override {
        return TYPE_MSE;
    }

    AutoDifferentialVar* loss(AutoDifferentialVar* inputs, NonAutoDifferentialVar*labels) override{
        return *(*inputs-*labels)**(*inputs-*labels);
    };
    AutoDifferentialVar* loss(AutoDifferentialVar* inputs, AutoDifferentialVar*labels) override{
        return *(*inputs-*labels)**(*inputs-*labels);
    };
    AutoDifferentialVar* loss(NonAutoDifferentialVar* inputs,AutoDifferentialVar*labels) override{
        return *(*inputs-*labels)**(*inputs-*labels);
    };
    NonAutoDifferentialVar* loss(NonAutoDifferentialVar* inputs, NonAutoDifferentialVar*labels) override{
        return *(*inputs-*labels)**(*inputs-*labels);
    };
private:
    MeanSquaredError()=default;
    friend class LossFunctionFactory;
};

#endif //TRANSFORMER_MEANSQUAREDERROR_H
