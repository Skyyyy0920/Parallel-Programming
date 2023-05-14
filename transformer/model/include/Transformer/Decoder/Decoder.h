//
// Created by Xuyao Wang on 3/13/2023.
//

#ifndef TRANSFORMER_DECODER_H
#define TRANSFORMER_DECODER_H

#include "DecoderBlock.h"
#include <vector>

class Decoder{
private:
    std::vector<DecoderBlock*>decoderBlock;
};

#endif //TRANSFORMER_DECODER_H
