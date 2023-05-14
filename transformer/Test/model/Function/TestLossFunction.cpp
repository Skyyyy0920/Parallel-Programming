//
// Created by Xuyao Wang on 5/12/2023.
//

#include "gtest/gtest.h"
#include "Function/LossFunction/LossFunctionFactory.h"
#include <iostream>

TEST(LossFunctionTest,MSEValidTest){
    LossFunction*lossFunction=LossFunctionFactory::getInstance().getMeanSquaredError();
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor t3({2, 2}, {0.01,0.16,0.04,0.09});
    auto* NADVar1=new NonAutoDifferentialVar(t1);
    auto* NADVar2=new NonAutoDifferentialVar(t2);

    auto * ADVar1=new AutoDifferentialVar({2,2});
    ADVar1->setData(t1);
    auto * ADVar2=new AutoDifferentialVar({2,2});
    ADVar2->setData(t2);

    auto result1=lossFunction->loss(ADVar1,ADVar2);
    auto result2=lossFunction->loss(ADVar1,NADVar2);
    auto result3=lossFunction->loss(NADVar1,ADVar2);
    auto result4=lossFunction->loss(NADVar1,NADVar2);
    EXPECT_EQ(result1->getData(),t3);
    EXPECT_EQ(result2->getData(),t3);
    EXPECT_EQ(result3->getData(),t3);
    EXPECT_EQ(result4->getData(),t3);
}

TEST(LossFunctionTest,MSEBackwardPropagationTest){
    LossFunction*lossFunction=LossFunctionFactory::getInstance().getMeanSquaredError();
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor t3({2, 2}, {0.1,0.16,0.04,0.9});
    auto* NADVar1=new NonAutoDifferentialVar(t1);
    auto* NADVar2=new NonAutoDifferentialVar(t2);
    auto* NADVar3=new NonAutoDifferentialVar(t3);

    auto * ADVar1=new AutoDifferentialVar({2,2});
    ADVar1->setData(t1);
    auto * ADVar2=new AutoDifferentialVar({2,2});
    ADVar2->setData(t2);

    auto label=NADVar3;
    auto output1=*(*(*ADVar1+*NADVar2)-*NADVar1)**NADVar1;
    // 0.1, 0.7, 0.24, 0.2          initial output
    // 0.1, 0.16, 0.04, 0.9         label
    // 0, 0.2916, 0.04, 0.49        initial loss
    auto loss1=lossFunction->loss(output1,label);
    for (int i = 0; i < 100000; ++i) {
        loss1->backward();
        output1=*(*ADVar1+*NADVar2)**NADVar1;
        loss1=lossFunction->loss(output1,label);

//        if (i%100==0)
//            loss1->print();
    }

    label->getData().print();
    output1->getData().print();
    EXPECT_EQ(output1->getData(),label->getData());

    auto output2=*(*ADVar1+*ADVar2)**ADVar1;
    auto loss2=lossFunction->loss(output2,label);
    for (int i = 0; i < 100000; ++i) {
        loss2->backward();
        output2=*(*ADVar1+*ADVar1)**ADVar1;
        loss2=lossFunction->loss(output2,label);
//        if (i%100==0)
//            loss2->print();
    }
    output2->getData().print();
    EXPECT_EQ(output2->getData(),label->getData());


    auto output3=*(*(*ADVar1-*ADVar2)**ADVar1)+*ADVar2;
    auto loss3=lossFunction->loss(output3,label);
    for (int i = 0; i < 100000; ++i) {
        loss3->backward();
        output3=*(*ADVar1-*ADVar2)**ADVar1;
        loss3=lossFunction->loss(output3,label);
//        if (i%100==0)
//            loss2->print();
    }
    output3->getData().print();
    EXPECT_EQ(output3->getData(),label->getData());
}

TEST(LossFunctionTest,MSESerialExecutionTime) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Your high-precision code to be measured goes here.
    LossFunction*lossFunction=LossFunctionFactory::getInstance().getMeanSquaredError();
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor t3({2, 2}, {0.1,0.16,0.04,0.9});
    auto* NADVar1=new NonAutoDifferentialVar(t1);
    auto* NADVar2=new NonAutoDifferentialVar(t2);
    auto* NADVar3=new NonAutoDifferentialVar(t3);

    auto * ADVar1=new AutoDifferentialVar({2,2});
    ADVar1->setData(t1);
    auto * ADVar2=new AutoDifferentialVar({2,2});
    ADVar2->setData(t2);

    auto label=NADVar3;
    auto output1=*(*(*ADVar1+*NADVar2)-*NADVar1)**NADVar1;

    auto output3=*(*(*ADVar1-*ADVar2)**ADVar1)+*ADVar2;
    auto loss3=lossFunction->loss(output3,label);
    for (int i = 0; i < 100000; ++i) {
        loss3->backward();
        output3=*(*ADVar1-*ADVar2)**ADVar1;
        loss3=lossFunction->loss(output3,label);
    }
    output3->getData().print();
    EXPECT_EQ(output3->getData(),label->getData());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;
}