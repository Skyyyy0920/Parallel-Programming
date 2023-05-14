//
// Created by Xuyao Wang on 3/14/2023.
//

#ifndef TRANSFORMER_NETWORK_H
#define TRANSFORMER_NETWORK_H

#include "Tensor/Tensor.h"

class Network{
public:
    virtual double forward(double input)=0;
};

#endif //TRANSFORMER_NETWORK_H
