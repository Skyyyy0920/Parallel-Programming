//
// Created by Xuyao Wang on 5/4/2023.
//

#ifndef TRANSFORMER_VAR_H
#define TRANSFORMER_VAR_H

#include <utility>

#include "nn/Tensor/Tensor.h"
#include "GradientInterface.h"
#include "nn/ComputationalGraph/Operator/Operator.h"
#include "nn/ComputationalGraph/Operator/OperatorFactory.h"

class Var{
protected:
    struct DataInfo{
        DataInfo(const Tensor& data,Operator*op):data(data),op(op){};
        DataInfo(const Tensor& data,Operator*op, std::vector<Var*>pre):data(data),op(op),predecessorNodes(std::move(pre)){};
        Operator* op= nullptr;           // Var的操作符
        Tensor data=Tensor();
        std::vector<Var*> predecessorNodes;
    }dataInfo;

protected:
    explicit Var(const Tensor& data):dataInfo(data, nullptr){};
    explicit Var(const std::vector<int>& shape):dataInfo(Tensor(shape), nullptr){};
    Var(const Tensor& data, Operator*op):dataInfo(data,op){};
    explicit Var(DataInfo dataInfo):dataInfo(std::move(dataInfo)){};
    virtual ~Var()= default;
public:
    virtual void print()=0;
public:
    virtual bool operator==(const Var& other) const;
    virtual bool operator!=(const Var& other) const;
    [[nodiscard]] virtual DataInfo Mul(const Var& other) const;
    [[nodiscard]] virtual DataInfo Add(const Var& other) const;
    [[nodiscard]] virtual DataInfo Sub(const Var& other) const;


public:
    [[nodiscard]] const Tensor &getData() const;
    void setData(const Tensor &tensor);
    [[nodiscard]] Operator *getOp() const;
    [[nodiscard]] const std::vector<Var*>& getPredecessorNodes() const;
    void setPredecessorNodes(const std::vector<Var*>& predecessorNodes);

};

#endif //TRANSFORMER_VAR_H
