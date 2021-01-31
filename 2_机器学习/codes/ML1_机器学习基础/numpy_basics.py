# numpy库的基本操作

import numpy as np
import time

# 1. 创建一个长度为10的一维全为0的ndarray对象，然后让第5个元素等于1
a = np.zeros((10,), int)
a[4] = 1
print(a)

# 2. 创建一个元素为从10到49的ndarray对象
a = np.arange(10, 50)
print(a)

# 3. 将第2题的所有元素位置反转
a = a[::-1]
print(a)

# 4. 使用np.random.random创建一个10*10的ndarray对象，并打印出最大最小元素
a = np.random.random((10, 10))
print(np.max(a), np.min(a))

# 5. 创建一个10*10的ndarray对象，且矩阵边界全为1，里面全为0
a = np.zeros((10, 10), dtype=int)
a[0, :] = 1
a[-1, :] = 1
a[:, 0] = 1
a[:, -1] = 1
print(a)

# 6. 创建一个每一行都是从0到4的5*5矩阵
a = np.arange(5).reshape(1, 5)
a = np.repeat(a, 5, axis=0)
print(a)

# 7. 创建一个范围在(0,1)之间的长度为12的等差数列
a = np.linspace(0, 1, 12)
print(a)

# 8. 创建一个长度为10的随机数组并排序
a = np.random.randint(0, 100, size=10)
a = np.sort(a)
print(a)

# 9. 创建一个长度为10的随机数组并将最大值替换为0
a = np.random.randint(0, 100, size=10)
a[np.argmax(a)] = 0
print(a)

# 10. 根据第3列来对一个5*5矩阵的行排序
a = np.random.randint(0, 100, (5, 5))
print(a)
x = np.argsort(a[:, 2])
print(x)
a = a[x]
print(a)

# 11. 给定一个4维矩阵，求最后两维的和
a = np.ones((3, 4, 5, 5))
s = np.sum(a, axis=(2, 3))  # axis里可以同时写多个维度

# 12. 给定数组[1, 2, 3, 4, 5]，如何得到在这个数组的每个元素之间插入3个0后的新数组
a = np.arange(1, 6).reshape(5, 1)
z = np.zeros((5, 3), dtype=int)
a = np.concatenate((a, z), axis=1).reshape(-1)[:-3]
print(a)

# 13. 给定一个二维矩阵，如何交换其中两行的元素
a = np.identity(10, int)
m = 3
n = 7
x = np.arange(10)
x[3] = 7
x[7] = 3
a = a[x]
print(a)

# 14. 创建一个10000000长度的随机数组，使用两种方法对其求三次方，并比较所用时间
a = np.random.randint(1, 10, 10000000)

time_start = time.time()
a1 = np.power(a, 3)
time_end = time.time()
print(a1)
print(time_end - time_start)

time_start = time.time()
a2 = a ** 3
time_end = time.time()
print(a2)
print(time_end - time_start)

# 15. 创建一个5*3随机矩阵和一个3*2随机矩阵，求矩阵积
a = np.random.randint(0, 10, (5, 3))
b = np.random.randint(0, 10, (3, 2))
print(a)
print(b)
c = np.dot(a, b)
print(c)

# 16. 矩阵的每一行的元素都减去该行的平均值
a = np.random.randint(0, 100, (3, 5))
a = a.astype(float)
ave = np.average(a, axis=1).reshape(3, 1)
ave = np.repeat(ave, 5, axis=1)
print(a)
print(ave)
a -= ave
print(a)

# 17. 打印出以下矩阵(要求使用np.zeros创建8*8的矩阵)：
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]
a = np.zeros((8, 8), dtype=int)
a[::2, 1::2] = 1
a[1::2, ::2] = 1
print(a)

# 18. 正则化一个5*5随机矩阵
a = np.random.randint(0, 100, (5, 5))
amin = np.min(a)
amax = np.max(a)
a = (a - amin) / (amax - amin)
print(a)

# 19. 将一个一维数组转化为二进制表示矩阵，例如[1,2,3]转化为
# [[0,0,1]
#  [0,1,0]
#  [0,1,1]]
I = np.array([1, 2, 3])
A = I.reshape(-1, 1)
B = 2 ** np.arange(3)
M = A & B
M[M != 0] = 1
print(M[:, ::-1])

# 20. 实现冒泡排序法
a = np.random.randint(0, 100, 10)
print(a)
len_a = len(a)
for i in range(0, len_a - 1):
    for j in range(0, len_a - 1 - i):
        if a[j] > a[j + 1]:
            t = a[j]
            a[j] = a[j + 1]
            a[j + 1] = t
print(a)
