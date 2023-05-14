//
// Created by Xuyao Wang on 4/9/2023.
//

#include "gtest/gtest.h"
#include "nn/Tensor/Tensor.h"

TEST(TensorTest, AddTest) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor result = t1 + t2;
    EXPECT_EQ(result, Tensor({2,2}, {0.5,1.4,0.6,0.5}));
}

TEST(TensorTest, SubTest) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor result = t1 - t2;
    EXPECT_EQ(result, Tensor({2,2}, {-0.1,-0.4,0.2,0.3}));
}

TEST(TensorTest, MulTest1) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor result = t1 * t2;
    EXPECT_EQ(result, Tensor({2,2}, {0.06,0.45,0.08,0.04}));
}

TEST(TensorTest, MulTest2) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor result = mul(t1, t2);
    EXPECT_EQ(result, Tensor({2,2}, {0.06,0.45,0.08,0.04}));
}

TEST(TensorTest, MatMulTest) {
    Tensor t1({4, 2}, {
        0.2,0.5,
        0.4,0.4,
        0.2,0.5,
        0.4,0.4});
    Tensor t2({2, 3}, {
        0.3,0.9,0.2,
        0.1,0.7,0.6});
    Tensor result = matmul(t1, t2);
    EXPECT_EQ(result, Tensor({4,3}, {0.11,0.53,0.34,0.16,
                                     0.64,0.32,0.11,0.53,
                                     0.34,0.16,0.64,0.32}));
}

TEST(TensorTest, DotTest) {
    Tensor t1({2}, {0.2,0.5});
    Tensor t2({2}, {0.3,0.9});
    double result= dot(t1, t2);
    EXPECT_EQ(result, 0.51);
}

TEST(TensorTest, DivTest) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor result = t1 / 0.2;
    EXPECT_EQ(result, Tensor({2,2}, {1.0,2.5,2.0,2.0}));
}

TEST(TensorTest, EQTest) {
    Tensor t({2, 2}, {0.2,0.5,0.4,0.4});
    EXPECT_EQ(t==Tensor({2, 2}, {0.2,0.5,0.4,0.4}), true);
    EXPECT_EQ(t==Tensor({2, 2}, {1.0,2.5,2.0,2.0}), false);
    EXPECT_EQ(t==Tensor({4, 1}, {0.2,0.5,0.4,0.4}), false);
}

TEST(TensorTest, NEQTest) {
    Tensor t({2, 2}, {0.2,0.5,0.4,0.4});
    EXPECT_EQ(t!=Tensor({2,2}, {1.0,2.5,2.0,2.0}), true);
    EXPECT_EQ(t!=Tensor({4, 1}, {0.2,0.5,0.4,0.4}), true);
    EXPECT_EQ(t!=Tensor({2, 2}, {0.2,0.5,0.4,0.4}), false);
}