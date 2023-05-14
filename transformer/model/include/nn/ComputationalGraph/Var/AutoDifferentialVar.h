//
// Created by Xuyao Wang on 5/4/2023.
//

#ifndef TRANSFORMER_AUTODIFFERENTIALVAR_H
#define TRANSFORMER_AUTODIFFERENTIALVAR_H

#include "Var.h"

class AutoDifferentialVar: public Var, public GradientInterface{
public:
    AutoDifferentialVar(const Tensor& data, Operator*op):Var(data,op), GradientInterface(1.0){};
    explicit AutoDifferentialVar(const std::vector<int>& shape):Var(Tensor(shape), nullptr), GradientInterface(1.0){};
    explicit AutoDifferentialVar(const DataInfo&dataInfo):Var(dataInfo),GradientInterface(1.0){};
private:
    explicit AutoDifferentialVar(const Var*var):Var(*var),GradientInterface(1.0){};
    explicit AutoDifferentialVar(const Var& var):Var(var),GradientInterface(1.0){};
public:
    void backward(const Tensor& gradient) override;
    void backward() override;
    bool operator==(const Var& other) const override;
    void print() override;
    bool isAutoDifferentiable() override;
public:

    AutoDifferentialVar& operator=(const Var& other){
        if (this == &other) {
            return *this;
        }
        this->dataInfo.data=other.getData();
        this->dataInfo.op=other.getOp();
        return *this;
    }
    AutoDifferentialVar* operator*(const Var& other) const{
        return new AutoDifferentialVar(Var::Mul(other));
    };
    AutoDifferentialVar* operator+(const Var& other) const{
        return new AutoDifferentialVar(Var::Add(other));
    }
    AutoDifferentialVar* operator-(const Var& other) const{
        return new AutoDifferentialVar(Var::Sub(other));
    }
};

#endif //TRANSFORMER_AUTODIFFERENTIALVAR_H
