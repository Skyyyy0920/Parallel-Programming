//
// Created by 86138 on 3/13/2023.
//

#ifndef TRANSFORMER_TRANSFORMER_H
#define TRANSFORMER_TRANSFORMER_H

#include "Encoder/EncoderBlock.h"
#include "Decoder/DecoderBlock.h"
#include "nn/Linear.h"
#include "nn/Softmax.h"
#include "Model.h"

class Transformer:public Model{
public:
    virtual void train() override;
private:
    EncoderBlock*encoderBlock;
    DecoderBlock*decoderBlock;
    Linear*linear;
    Softmax*softmax;
};

#endif //TRANSFORMER_TRANSFORMER_H
