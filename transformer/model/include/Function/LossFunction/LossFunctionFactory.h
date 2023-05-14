//
// Created by Xuyao Wang on 5/11/2023.
//

#ifndef TRANSFORMER_LOSSFUNCTIONFACTORY_H
#define TRANSFORMER_LOSSFUNCTIONFACTORY_H

#include "MeanSquaredError.h"

class LossFunctionFactory{
public:
    static LossFunctionFactory& getInstance(){
        static LossFunctionFactory instance;
        return instance;
    }
    LossFunction*getMeanSquaredError(){
        if (!mse){
            mse=new MeanSquaredError();
        }
        return mse;
    }
private:
    MeanSquaredError*mse= nullptr;
};

#endif //TRANSFORMER_LOSSFUNCTIONFACTORY_H
