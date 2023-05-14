//
// Created by Xuyao Wang on 3/13/2023.
//

#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include "EncoderBlock.h"
#include <vector>

class Encoder{
    std::vector<EncoderBlock*>encoderBlock;
};

#endif //TRANSFORMER_ENCODER_H
