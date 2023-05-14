//
// Created by Xuyao Wang on 5/3/2023.
//

#include <utility>
#include <pthread.h>
#include <thread>

#include "nn/Tensor/Tensor.h"

void Tensor::print() const {
    std::cout << "Tensor with shape ({";
    for (int i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i != shape_.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "}):" << std::endl;

    print(0, std::vector<int>(shape_.size(), 0));
    std::cout << std::endl;
}

Tensor::Tensor(const std::vector<int>& shape, std::vector<double> data)
        : shape_(shape),size_(1), data_(std::move(data))
{
    for (int dim : shape) {
        size_ *= dim;
    }
}

Tensor::Tensor(const std::vector<int>& shape, double data[])
        : shape_(shape),size_(1)
{
    assert(data!= nullptr);
    for (int dim : shape) {
        size_ *= dim;
    }
    for (int i = 0; i < size_; ++i) {
        data_.push_back(data[i]);
    }
}

Tensor::Tensor(const std::vector<int>& shape, double data)
        : shape_(shape),size_(1)
{
    for (int dim : shape) {
        size_ *= dim;
    }
    data_.resize(size_);
    for (int i = 0; i < size_; ++i) {
        data_[i]=data;
    }
}

Tensor::Tensor(std::vector<int> shape)
        : shape_(shape), size_(1)
{
    for (int dim : shape) {
        size_ *= dim;
    }
    // 为Tensor对象分配内存
    data_.resize(size_);

    // 使用std::random_device获取随机数种子
    std::random_device rd;
    // 使用std::mt19937作为随机数生成器
    std::mt19937 gen(rd());
    // 使用std::normal_distribution生成0-1正态分布的随机数
    std::normal_distribution<double> distribution(0.5, 0.1);

    // 将0-1正态分布的随机数存储在vector中
    for (int i = 0; i < size_; ++i) {
        double value = distribution(gen);
        data_[i] = value;
    }
}

Tensor::Tensor (const Tensor& other) {
    shape_ = other.shape_;
    data_ = other.data_;
    size_ = other.size_;
}

int Tensor::indexToOffset(std::vector<int> index) const {
    if (index.size() != shape_.size()) {
        throw std::runtime_error("Incorrect number of dimensions.");
    }
    // 计算偏移量
    int offset = 0;
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0; i--) {
        if (index[i] < 0 || index[i] >= shape_[i]) {
            throw std::runtime_error("Index out of range.");
        }
        offset += index[i] * stride;
        stride *= shape_[i];
    }

    return offset;
}

void Tensor::print(int dim, std::vector<int> indices) const {
    if (dim == shape_.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < shape_[dim]; i++) {
            if (i > 0) {
                std::cout << ", ";
            }
            indices[dim] = i;
            std::cout << data_[indexToOffset(indices)];
        }
        std::cout << "]";
    } else {
        std::cout << "[";
        for (int i = 0; i < shape_[dim]; i++) {
            if (i > 0) {
                std::cout << ",\n ";
            }
            indices[dim] = i;
            print(dim + 1, indices);
        }
        std::cout << "]";
    }
}

// 定义 dot 函数
double dot(const Tensor& a, const Tensor& b) {
    // 判断两个向量的长度是否一致
    if (a.size_ != b.size_ || a.shape_.size() != 1 || b.shape_.size() != 1) {
        throw std::invalid_argument("Invalid input for dot product.");
    }

    double result = 0;

    // 计算点乘结果
    for (int i = 0; i < a.size_; ++i) {
        result += a.data_[i] * b.data_[i];
    }

    return result;
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(Tensor::is_valid_mm(A, B));

    Tensor C({A.shape_[0], B.shape_[1]});

    for (int i = 0; i < A.shape_[0]; ++i) {
        for (int j = 0; j < B.shape_[1]; ++j) {
            for (int k = 0; k < A.shape_[1]; ++k) {
                C({i, j}) += A({i, k}) * B({k, j});
            }
        }
    }

    return C;
}

Tensor mul(const Tensor& A, const Tensor& B) {
    // 检查shape是否相同
    assert(A.shape_ == B.shape_);

    // 创建新的Tensor
    Tensor result(A.shape_);

    // 对应元素相乘
    for (int i = 0; i < A.size_; i++) {
        result[i] = A.data_[i] * B.data_[i];
    }

    return result;
}


bool Tensor::is_valid_mm(const Tensor& A, const Tensor& B) {
    if (A.shape_.size() == 2 && B.shape_.size() == 2){
        return A.shape_[1] == B.shape_[0];
    } else if (A.shape_.size() == 1 && B.shape_.size() == 2){
        return A.shape_[0] == B.shape_[0];
    } else if (A.shape_.size() == 2 && B.shape_.size() == 1){
        return A.shape_[1] == B.shape_[0];
    } else{
        return false;
    }
}

bool Tensor::operator==(const Tensor& other) const {
    // 进行逐个元素的比较
    if (this->shape_ != other.shape_) {
        return false;
    }

    for (int i = 0; i < this->size_; ++i) {
        if (abs(this->data_[i] - other.data_[i])>1e-7) {
            return false;
        }
    }
    return true;
}

bool Tensor::operator!=(const Tensor& other) const {
    return !(*this==other);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    shape_ = other.shape_;
    data_ = other.data_;
    size_ = other.size_;
    return *this;
}

Tensor Tensor::T() const {
    // 创建一个新的形状为(n, m)的Tensor对象，其中n为原始Tensor对象的列数，m为原始Tensor对象的行数
    std::vector<int> newShape = {shape_[1], shape_[0]};
    Tensor result(newShape);

    // 将原始Tensor对象的每个元素赋值给转置后的Tensor对象的相应位置
    for (int i = 0; i < shape_[0]; i++) {
        for (int j = 0; j < shape_[1]; j++) {
            std::vector<int> index = {j, i};
            result(index) = (*this)(index);
        }
    }

    return result;
}

int Tensor::dim(int d) const{
    assert(d<shape_.size());
    return shape_[d];
}

double& Tensor::operator[](int i) { return data_[i]; }
const double& Tensor::operator[](int i) const { return data_[i]; }

double& Tensor::operator()(std::vector<int> index) {
    int i = indexToOffset(index);
    return data_[i];
}

const double& Tensor::operator()(std::vector<int> index) const {
    int i = indexToOffset(index);
    return data_[i];
}

// 重载 Tensor 加法运算符，支持并行计算
Tensor Tensor::operator+(const Tensor& other) const {
    // 检查两个Tensor对象的形状是否相同
    if (shape_ != other.shape_) {
        throw std::runtime_error("Cannot add two tensors with different shapes.");
    }
    // 创建一个新的Tensor对象
    Tensor result(shape_);
    // 对两个Tensor对象中的每一个元素执行加法运算
    auto add_func = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            result.data_[i] = data_[i] + other.data_[i];
        }
    };
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = result.data_.size() / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? result.data_.size() : (i + 1) * chunk_size;
        threads.emplace_back(add_func, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

// 重载运算符-
Tensor Tensor::operator-(const Tensor& other) const {
    // 检查两个Tensor对象的形状是否相同
    if (shape_ != other.shape_) {
        throw std::runtime_error("Cannot add two tensors with different shapes.");
    }
    // 创建一个新的Tensor对象
    Tensor result(shape_);
    // 对两个Tensor对象中的每一个元素执行减法运算
    auto add_func = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            result.data_[i] = data_[i] - other.data_[i];
        }
    };
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = result.data_.size() / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? result.data_.size() : (i + 1) * chunk_size;
        threads.emplace_back(add_func, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

// 重载运算符*
Tensor Tensor::operator*(const Tensor& other) const {
    // 检查两个Tensor对象的形状是否相同
    if (shape_ != other.shape_) {
        throw std::runtime_error("Cannot add two tensors with different shapes.");
    }
    // 创建一个新的Tensor对象
    Tensor result(shape_);
    // 对两个Tensor对象中的每一个元素执行乘法运算
    auto add_func = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            result.data_[i] = data_[i] * other.data_[i];
        }
    };
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = result.data_.size() / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? result.data_.size() : (i + 1) * chunk_size;
        threads.emplace_back(add_func, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

// 重载运算符/
Tensor Tensor::operator/(const Tensor& other) const {
    // 检查两个Tensor对象的形状是否相同
    if (shape_ != other.shape_) {
        throw std::runtime_error("Cannot add two tensors with different shapes.");
    }
    // 创建一个新的Tensor对象
    Tensor result(shape_);
    // 对两个Tensor对象中的每一个元素执行除法运算
    auto add_func = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            assert(other.data_[i]!=0);
            result.data_[i] = data_[i] / other.data_[i];
        }
    };
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = result.data_.size() / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? result.data_.size() : (i + 1) * chunk_size;
        threads.emplace_back(add_func, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

// 重载 Tensor 乘法运算符，返回 Tensor 与 double 相乘的结果
Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape_);
    auto mul_func = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            result.data_[i] = data_[i] * scalar;
        }
    };
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = result.data_.size() / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? result.data_.size() : (i + 1) * chunk_size;
        threads.emplace_back(mul_func, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

// 重载 Tensor 除法运算符，返回 Tensor 与 double 相除的结果
Tensor Tensor::operator/(double scalar) const {
    assert(scalar!=0);
    Tensor result(shape_);
    auto mul_func = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            result.data_[i] = data_[i] / scalar;
        }
    };
    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = result.data_.size() / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? result.data_.size() : (i + 1) * chunk_size;
        threads.emplace_back(mul_func, start, end);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result;
}

// 重载 Tensor 乘法运算符，返回 Tensor 与 int 相乘的结果
Tensor Tensor::operator*(int scalar) const {
    Tensor result(shape_);

    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

// 重载 Tensor 除法运算符，返回 Tensor 与 int 相除的结果
Tensor Tensor::operator/(int scalar) const {
    assert(scalar!=0.0);
    Tensor result(shape_);

    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i] / scalar;
    }

    return result;
}

const std::vector<int> &Tensor::getShape() const {
    return shape_;
}

void Tensor::setShape(const std::vector<int> &shape) {
    shape_ = shape;
}

const std::vector<double> &Tensor::getData() const {
    return data_;
}

void Tensor::setData(const std::vector<double> &data) {
    data_ = data;
}

int Tensor::getSize() const {
    return size_;
}

void Tensor::setSize(int size) {
    size_ = size;
}

void Tensor::reset() {
    std::fill(data_.begin(), data_.end(),0.0);
}

Tensor Tensor::operator-() const {
    Tensor result(shape_);

    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = -1*data_[i];
    }

    return result;
}

Tensor Tensor::operator+(double scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i]+scalar;
    }
    return result;
}

Tensor Tensor::operator-(double scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i]-scalar;
    }
    return result;
}

Tensor Tensor::operator+(int scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i]+scalar;
    }
    return result;
}

Tensor Tensor::operator-(int scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < data_.size(); i++) {
        result.data_[i] = data_[i]-scalar;
    }
    return result;
}
