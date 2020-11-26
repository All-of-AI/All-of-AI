# 利用sympy库求积分

import sympy

# 使用integrate()函数来计算定积分和不定积分

x = sympy.Symbol('x')
f = 2 * x
res = sympy.integrate(f, (x, 0, 1))  # 定积分，传入一个元组
print(res)  # 1

f = sympy.E ** x + 2 * x
res = sympy.integrate(f, x)  # 不定积分，只传入变量
print(res)  # x**2 + exp(x)
