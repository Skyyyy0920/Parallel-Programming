//
// Created by 86138 on 5/11/2023.
//

#ifndef TRANSFORMER_LOSSFUNCTION_H
#define TRANSFORMER_LOSSFUNCTION_H

#include "nn/ComputationalGraph/Var/AutoDifferentialVar.h"
#include "nn/ComputationalGraph/Var/NonAutoDifferentialVar.h"

class LossFunction{
protected:
    enum LossFunctionType {
        TYPE_MSE,
        TYPE_CE
    };

    LossFunction() = default;
public:
    [[nodiscard]] virtual LossFunctionType getType() const = 0;
    virtual AutoDifferentialVar* loss(AutoDifferentialVar* inputs, NonAutoDifferentialVar*labels)=0;
    virtual AutoDifferentialVar* loss(AutoDifferentialVar* inputs, AutoDifferentialVar*labels)=0;
    virtual AutoDifferentialVar* loss(NonAutoDifferentialVar* inputs, AutoDifferentialVar*labels)=0;
    virtual NonAutoDifferentialVar* loss(NonAutoDifferentialVar* inputs, NonAutoDifferentialVar*labels)=0;
public:
    static LossFunctionType getMSEType(){return TYPE_MSE;}
    static LossFunctionType getCEType(){return TYPE_CE;}


};

#endif //TRANSFORMER_LOSSFUNCTION_H
