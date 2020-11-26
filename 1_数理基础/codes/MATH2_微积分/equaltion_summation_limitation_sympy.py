# 使用sympy库实现表达式求值，方程组求解，数列求和以及求极限

import sympy

# --------------------表达式求值--------------------
# 定义单变量函数
x = sympy.Symbol('x')
fx = 5 * x + 4
# 使用evalf()函数来对变量传值
res = fx.evalf(subs={x: 6})
print(res)

# 定义多变量函数
x = sympy.Symbol('x')
y = sympy.Symbol('y')
fxy = x * x + y * y
res = fxy.evalf(subs={x: 2, y: 1})
print(res)

# --------------------方程组求解--------------------
# 单个方程的情况
x = sympy.Symbol('x')
y = sympy.Symbol('y')
fx = 3 * x + 9
print(sympy.solve(fx, x))  # [-3]

# 无穷多个解的情况，可以得到变量之间的关系
x = sympy.Symbol('x')
y = sympy.Symbol('y')
fx = x * 3 + y ** 2
print(sympy.solve(fx, x, y))  # [{x: -y**2/3}]

# 方程组的情况
x = sympy.Symbol('x')
y = sympy.Symbol('y')
f1 = x + y - 3
f2 = x - y + 5
print(sympy.solve([f1, f2], [x, y]))  # {x: -1, y: 4}

# --------------------数列求和--------------------
n = sympy.Symbol('n')
f = n
res = sympy.summation(f, (n, 1, 100))
print(res)  # 5050

# 求解带有求和形式的方程
x = sympy.Symbol('x')
i = sympy.Symbol('i')
f = sympy.summation(x, (i, 1, 5)) + 10 * x - 15
res = sympy.solve(f, x)
print(res)  # [1]

# --------------------求极限--------------------
x = sympy.Symbol('x')
f = (1 + x) ** (1 / x)
lim = sympy.limit(f, x, 0)
print(lim)  # E
