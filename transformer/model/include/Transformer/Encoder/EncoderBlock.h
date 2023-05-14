//
// Created by Xuyao Wang on 3/13/2023.
//

#ifndef TRANSFORMER_ENCODERBLOCK_H
#define TRANSFORMER_ENCODERBLOCK_H

#include "nn/AddNorm.h"
#include "nn/FeedForward.h"
#include "Transformer/SelfAttention/MultiHeadAttention.h"

class EncoderBlock {
private:
    MultiHeadAttention* multiHeadAttention;
    AddNorm* addNorm1;
    FeedForward* feedForward;
    AddNorm* addNorm2;

};


#endif //TRANSFORMER_ENCODERBLOCK_H
