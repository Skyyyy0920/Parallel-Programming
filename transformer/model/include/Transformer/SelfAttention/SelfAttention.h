//
// Created by Xuyao Wang on 3/13/2023.
//

#ifndef TRANSFORMER_SELFATTENTION_H
#define TRANSFORMER_SELFATTENTION_H

#include "Config.h"
typedef long double LD;

class SelfAttention{
private:
    LD** WQ;
    LD** WK;
    LD** WV;
};

#endif //TRANSFORMER_SELFATTENTION_H
