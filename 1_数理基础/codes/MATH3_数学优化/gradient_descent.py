# 实现梯度下降算法逼近一个函数的最小值

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-80, 80, 5000)


def f(x):
    # 目标函数
    return x ** 2 + 5 * x + 2


def df(x):
    # 目标函数的导数
    return 2 * x + 5


alpha = 0.1  # 学习率

plt.plot(x, f(x))

init_x = 76  # x的初始值
init_value = f(init_x)

last_x = init_x
last_value = init_value

xs = [init_x]
values = [init_value]

# 10000次迭代
for i in range(10000):
    now_x = last_x - alpha * df(last_x)
    now_value = f(now_x)

    if abs(now_value - last_value) < 1e-8:  # 当某一步优化进行不明显时，终止优化
        break

    last_x = now_x
    last_value = now_value

    print('The {}th step, x={}, f(x)={}'.format(i + 1, last_x, last_value))

    xs.append(last_x)
    values.append(last_value)

plt.plot(xs, values, color='red')
plt.show()
