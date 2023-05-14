//
// Created by Xuyao Wang on 5/4/2023.
//
//
#include "gtest/gtest.h"
#include "nn/ComputationalGraph/Var/AutoDifferentialVar.h"
#include "nn/ComputationalGraph/Var/NonAutoDifferentialVar.h"

TEST(VarTest, VarAddTest) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor t3({2, 2}, {0.5,1.4,0.6,0.5});
    auto* NADVar1=new NonAutoDifferentialVar(t1);
    auto* NADVar2=new NonAutoDifferentialVar(t2);
    auto* NADVar3=new NonAutoDifferentialVar(t3);
    NonAutoDifferentialVar* NADVarAddResult=*NADVar1+*NADVar2;
    EXPECT_EQ(*NADVarAddResult,*NADVar3);
    EXPECT_EQ(NADVarAddResult->getData(),NADVar3->getData());
    EXPECT_EQ(NADVarAddResult->isAutoDifferentiable(), false);
    EXPECT_EQ(NADVarAddResult->getOp()->getType(),Operator::getAddType());

    auto * ADVar1=new AutoDifferentialVar({2,2});
    ADVar1->setData(t1);
    auto * ADVar2=new AutoDifferentialVar({2,2});
    ADVar2->setData(t2);
    auto * ADVarAddResult=*ADVar1+*ADVar2;
    EXPECT_EQ(ADVarAddResult->getData(),t3);
    EXPECT_EQ(ADVarAddResult->getOp()->getType(),Operator::getAddType());
    EXPECT_EQ(ADVarAddResult->isAutoDifferentiable(), true);

    // TEST AD+NAD
    Var* AD_NADVarAddResult1=*ADVar1+*NADVar2;
    auto * AD_NADVarAddResult2=*ADVar1+*NADVar2;
    EXPECT_EQ(AD_NADVarAddResult1->getData(),t3);
    EXPECT_EQ(AD_NADVarAddResult2->getData(),t3);
    EXPECT_EQ(AD_NADVarAddResult1->getOp()->getType(),Operator::getAddType());
    EXPECT_EQ(AD_NADVarAddResult2->getOp()->getType(),Operator::getAddType());
    EXPECT_EQ(dynamic_cast<GradientInterface*>(AD_NADVarAddResult1)->isAutoDifferentiable(), true);
    EXPECT_EQ(dynamic_cast<GradientInterface*>(AD_NADVarAddResult2)->isAutoDifferentiable(), true);
    EXPECT_EQ(*AD_NADVarAddResult1,*AD_NADVarAddResult2);

    auto AD_NADVarAddResult3=*ADVar1+*NADVar2;
    EXPECT_EQ(dynamic_cast<GradientInterface*>(AD_NADVarAddResult3)->isAutoDifferentiable(), true);

    // TEST NAD+AD
    Var* NAD_ADVarAddResult1=*NADVar2+*ADVar1;
    auto NAD_ADVarAddResult2=*NADVar2+*ADVar1;
    EXPECT_EQ(dynamic_cast<GradientInterface*>(NAD_ADVarAddResult1)->isAutoDifferentiable(), true);
    EXPECT_EQ(dynamic_cast<GradientInterface*>(NAD_ADVarAddResult2)->isAutoDifferentiable(), true);
    EXPECT_EQ(*NAD_ADVarAddResult1,*NAD_ADVarAddResult2);
}


TEST(VarTest, VarMulTest) {
    Tensor t1({2, 2}, {0.2,0.5,0.4,0.4});
    Tensor t2({2, 2}, {0.3,0.9,0.2,0.1});
    Tensor t3({2, 2}, {0.06,0.45,0.08,0.04});
    auto NADVar1 = new NonAutoDifferentialVar(t1);
    auto NADVar2 = new NonAutoDifferentialVar(t2);
    auto NADVar3 = new NonAutoDifferentialVar(t3);
    Var* NADVarAddResult=*NADVar1**NADVar2;
    EXPECT_EQ(*NADVarAddResult,*NADVar3);

    auto ADVar1 = new AutoDifferentialVar({2,2});
    ADVar1->setData(t1);
    auto ADVar2 = new AutoDifferentialVar({2,2});
    ADVar2->setData(t2);
    auto ADVarAddResult=*ADVar1**ADVar2;
    EXPECT_EQ(ADVarAddResult->getData(),t3);
    EXPECT_EQ(ADVarAddResult->getOp()->getType(),Operator::getMulType());
    EXPECT_EQ(ADVarAddResult->isAutoDifferentiable(), true);

    auto AD_NADVarAddResult1=*ADVar1**NADVar2;
    auto AD_NADVarAddResult2=*ADVar1**NADVar2;
    EXPECT_EQ(AD_NADVarAddResult1->getData(),t3);
    EXPECT_EQ(AD_NADVarAddResult2->getData(),t3);
    EXPECT_EQ(AD_NADVarAddResult1->getOp()->getType(),Operator::getMulType());
    EXPECT_EQ(AD_NADVarAddResult2->getOp()->getType(),Operator::getMulType());
    EXPECT_EQ(AD_NADVarAddResult1->isAutoDifferentiable(), true);
    EXPECT_EQ(AD_NADVarAddResult2->isAutoDifferentiable(), true);
}

