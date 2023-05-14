//
// Created by Xuyao Wang on 3/14/2023.
//

#ifndef TRANSFORMER_TENSOR_H
#define TRANSFORMER_TENSOR_H

#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <random>

class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<int>& shape, std::vector<double> data);
    Tensor(const std::vector<int>& shape, double data[]);
    Tensor(const std::vector<int>& shape, double data);
    explicit Tensor(std::vector<int> shape);
    Tensor(const Tensor& other);

    double& operator[](int i);
    const double& operator[](int i) const;

    // 重载运算符()，用于获取或设置Tensor对象中的元素值
    double& operator()(std::vector<int> index);
    const double& operator()(std::vector<int> index) const;

    Tensor operator-() const;

    // 重载运算符+-，用于实现Tensor对象之间的加减运算
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;

    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;

    Tensor operator+(int scalar) const;
    Tensor operator-(int scalar) const;

    // 重载运算符* and /，用于实现Tensor对象之间的对位乘法运算
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;

    Tensor operator*(int scalar) const;
    Tensor operator/(int scalar) const;

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    Tensor& operator=(const Tensor& other);

    friend Tensor matmul(const Tensor& A, const Tensor& B); // 二维Tensor矩阵乘法
    friend Tensor mul(const Tensor& A, const Tensor& B) ;   // Tensor对应元素相乘
    friend double dot(const Tensor& a, const Tensor& b);    // 向量点乘

    Tensor T() const;   //矩阵转置
    void print() const;
    void reset();

    friend class MatmulOperation;

private:
    std::vector<int> shape_;    // Tensor对象的形状
    std::vector<double> data_;  // Tensor对象的数据
    int size_;                  // Tensor对象的大小
public:
    [[nodiscard]] const std::vector<int> &getShape() const;
    void setShape(const std::vector<int> &shape);
    [[nodiscard]] const std::vector<double> &getData() const;
    void setData(const std::vector<double> &data);
    [[nodiscard]] int getSize() const;
    void setSize(int size);

    // 获取shape的某一维度大小
    int dim(int d) const;

private:
    static bool is_valid_mm(const Tensor& A, const Tensor& B);
    void print(int dim, std::vector<int> indices) const;
    int indexToOffset(std::vector<int> index) const;        // 将索引转换为偏移量
};

#endif //TRANSFORMER_TENSOR_H
