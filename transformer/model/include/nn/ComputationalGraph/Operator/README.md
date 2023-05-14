# Operator

for the method

```c++
std::vector<Tensor> backward(const Tensor& grad_output) = 0;
```

the input is $ \frac{\partial L}{\partial z}$

## AddOperator

for the equation below:

$$
L=f(z)\\
z=x+y\\
$$

we return pair$\{ \frac{\partial L}{\partial z} \frac{\partial z}{\partial x}, \frac{\partial L}{\partial z} \frac{\partial z}{\partial y}\}$

if both x and y are variable, $\frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$ are 1, the return pair is $\{\frac{\partial L}{\partial z},\frac{\partial L}{\partial z}\}$

if x is constant, the return pair is $\{0,\frac{\partial L}{\partial z}\}$

if y is constant, the return pair is $\{\frac{\partial L}{\partial z},0\}$

## MulOperator

for the equation below:

$$
L=f(z)\\
z=x*y\\
$$

we return pair$\{ \frac{\partial L}{\partial z} \frac{\partial z}{\partial x}, \frac{\partial L}{\partial z} \frac{\partial z}{\partial y}\}$

if both x and y are variable, the return pair is $\{\frac{\partial L}{\partial z}y,\frac{\partial L}{\partial z}x\}$

if x is constant, the return pair is $\{0,\frac{\partial L}{\partial z}x\}$

if y is constant, the return pair is $\{\frac{\partial L}{\partial z}y,0\}$

## DivOperator

for the equation below:

$$
L=f(z)\\
z=\frac{x}{y}\\
$$

we return pair$\{ \frac{\partial L}{\partial z} \frac{\partial z}{\partial x}, \frac{\partial L}{\partial z} \frac{\partial z}{\partial y}\}$

if both x and y are variable, the return pair is $\{\frac{\partial L}{\partial z} \frac{1}{y},-\frac{\partial L}{\partial z}\frac{x}{y^2}\}$

if x is constant, the return pair is $\{0,-\frac{\partial L}{\partial z}\frac{x}{y^2}\}$

if y is constant, the return pair is $\{\frac{\partial L}{\partial z} \frac{1}{y},0\}$
