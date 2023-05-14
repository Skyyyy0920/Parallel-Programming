//
// Created by Xuyao Wang on 3/13/2023.
//

#ifndef TRANSFORMER_MULTIHEADATTENTION_H
#define TRANSFORMER_MULTIHEADATTENTION_H

#include "SelfAttention.h"
#include "vector"

class MultiHeadAttention{
private:
    std::vector<SelfAttention*>selfAttention;
};

#endif //TRANSFORMER_MULTIHEADATTENTION_H
