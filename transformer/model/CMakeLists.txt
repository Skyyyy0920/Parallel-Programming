project(model)

# 将头文件添加到库的包含目录中
include_directories(include)

# 添加源文件
add_library(model STATIC
        src/nn/Tensor/Tensor.cpp
        src/nn/AddNorm.cpp
        src/Transformer/Decoder/Decoder.cpp
        src/Transformer/Decoder/DecoderBlock.cpp
        src/Transformer/Encoder/Encoder.cpp
        src/Transformer/Encoder/EncoderBlock.cpp
        src/nn/FeedForward.cpp
        src/Transformer/SelfAttention/MultiHeadAttention.cpp
        src/Transformer/SelfAttention/SelfAttention.cpp
        src/Transformer/Transformer.cpp
        src/nn/ComputationalGraph/Var/Var.cpp
        src/nn/ComputationalGraph/Var/AutoDifferentialVar.cpp
        src/nn/ComputationalGraph/Var/NonAutoDifferentialVar.cpp
        src/nn/ComputationalGraph/Operator/AddOperator.cpp
        src/nn/ComputationalGraph/Operator/MulOperator.cpp include/nn/ComputationalGraph/Var/VarUtility.h include/Function/LossFunction/LossFunction.h include/Function/LossFunction/LossFunctionFactory.h include/Function/LossFunction/MeanSquaredError.h include/nn/ComputationalGraph/Operator/SubOperator.h src/nn/ComputationalGraph/Operator/SubOperator.cpp)

target_include_directories(model
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include)
