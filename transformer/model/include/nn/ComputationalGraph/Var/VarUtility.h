//
// Created by Xuyao Wang on 5/6/2023.
//

#ifndef TRANSFORMER_VARUTILITY_H
#define TRANSFORMER_VARUTILITY_H

#include "AutoDifferentialVar.h"
#include "NonAutoDifferentialVar.h"

class VarUtility{
private:
    VarUtility()=default;
public:
    static Var*createVar(const Tensor&data, Operator* op,bool isADV){
        return isADV?(Var*)new AutoDifferentialVar(data,op):new NonAutoDifferentialVar(data,op);
    }
};

#endif //TRANSFORMER_VARUTILITY_H
