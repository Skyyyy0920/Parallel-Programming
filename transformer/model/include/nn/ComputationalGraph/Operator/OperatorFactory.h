//
// Created by Xuyao Wang on 5/6/2023.
//

#ifndef TRANSFORMER_OPERATORFACTORY_H
#define TRANSFORMER_OPERATORFACTORY_H

#include "Operator.h"
#include "AddOperator.h"
#include "SubOperator.h"
#include "MulOperator.h"

class OperatorFactory{
public:
    static OperatorFactory& getInstance() {
        static OperatorFactory instance;
        return instance;
    }

    Operator* getAddOperator() {
        if (!addOperator){
            addOperator=new AddOperator();
        }
        return addOperator;
    }

    Operator* getSubOperator() {
        if (!subOperator){
            subOperator=new SubOperator();
        }
        return subOperator;
    }

    Operator* getMulOperator() {
        if (!mulOperator){
            mulOperator=new MulOperator();
        }
        return mulOperator;
    }
private:
    OperatorFactory() = default;
    OperatorFactory(const OperatorFactory&) = delete;
    OperatorFactory& operator=(const OperatorFactory&) = delete;
    ~OperatorFactory() {
        delete addOperator;
        delete mulOperator;
    }
private:
    AddOperator* addOperator= nullptr;
    SubOperator* subOperator= nullptr;
    MulOperator* mulOperator= nullptr;
};

#endif //TRANSFORMER_OPERATORFACTORY_H
