# 利用sympy库实现自动微分

import sympy

# 使用diff()函数来计算导数
x = sympy.Symbol('x')
f1 = 2 * x ** 4 + 3 * x + 6
f1_ = sympy.diff(f1, x)
print(f1_)

f2 = sympy.sin(x)
f2_ = sympy.diff(f2, x)
print(f2_)

# 计算偏导数
y = sympy.Symbol('y')
f3 = 2 * x ** 2 + 3 * y ** 4 + 2 * y
# 分别计算函数f3对于变量x和y的偏导数
f3_x = sympy.diff(f3, x)
f3_y = sympy.diff(f3, y)
print('partial derivative of x: ', f3_x)
print('partial derivative of y: ', f3_y)

# 链式法则
x = sympy.Symbol('x')
u = sympy.sin(x)
v = u ** 2
print(sympy.diff(v, x))
