//
// Created by 86138 on 5/4/2023.
//

#ifndef TRANSFORMER_NONAUTODIFFERENTIALVAR_H
#define TRANSFORMER_NONAUTODIFFERENTIALVAR_H

#include "AutoDifferentialVar.h"

class NonAutoDifferentialVar:public Var, public GradientInterface{
public:
    NonAutoDifferentialVar(const Tensor& data, Operator*op):Var(data,op), GradientInterface(0.0){};
    explicit NonAutoDifferentialVar(const Tensor& data):Var(data,nullptr), GradientInterface(0.0){};
private:
    explicit NonAutoDifferentialVar(const DataInfo&dataInfo):Var(dataInfo), GradientInterface(0.0){};
public:
    bool operator==(const Var& other) const override;
    bool operator!=(const Var& other) const override;
    void backward(const Tensor& gradient) override;
    void backward() override;
    void print() override;
    bool isAutoDifferentiable() override;
public:
    NonAutoDifferentialVar& operator=(const Var& other){
        if (this == &other) {
            return *this;
        }
        this->dataInfo.data=other.getData();
        this->dataInfo.op=other.getOp();
        return *this;
    }
    NonAutoDifferentialVar* operator*(const NonAutoDifferentialVar& other) const{
        return new NonAutoDifferentialVar(Var::Mul(other));
    };
    AutoDifferentialVar* operator*(const AutoDifferentialVar& other) const{
        return new AutoDifferentialVar(Var::Mul(other));
    };

    NonAutoDifferentialVar* operator+(const NonAutoDifferentialVar& other) const{
        return new NonAutoDifferentialVar(Var::Add(other));
    }
    AutoDifferentialVar* operator+(const AutoDifferentialVar& other) const{
        return new AutoDifferentialVar(Var::Add(other));
    }

    NonAutoDifferentialVar* operator-(const NonAutoDifferentialVar& other) const{
        return new NonAutoDifferentialVar(Var::Sub(other));
    }
    AutoDifferentialVar* operator-(const AutoDifferentialVar& other) const{
        DataInfo info=Var::Sub(other);
        return new AutoDifferentialVar(Var::Sub(other));
    }
};

#endif //TRANSFORMER_NONAUTODIFFERENTIALVAR_H
