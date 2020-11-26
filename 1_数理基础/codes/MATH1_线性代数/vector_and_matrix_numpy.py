# 使用numpy实现线性代数中向量和矩阵的基本操作

import numpy as np

# --------------------向量--------------------

# 向量的定义
v_1 = np.array([1, 2, 3, 4, 5])
v_2 = np.array([5.6, 4.6, 3.6, 2.6, 1.6])

# 向量加法
v_a = np.add(v_1, v_2)
print(v_a, v_a.shape)  # 结果为向量

# 向量的内积
v_i = np.inner(v_1, v_2)
print(v_i, v_i.shape)  # 结果为标量

# 向量的外积
v_o = np.outer(v_1, v_2)
print(v_o, v_o.shape)  # 结果为矩阵

# L2正则化
v_1_2 = v_1 * v_1  # '*'运算符代表对应位置相乘
sum_l2 = np.sum(v_1_2)
l2 = np.sqrt(sum_l2)
print(l2)

# --------------------矩阵--------------------

# 矩阵的定义
A = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]  # (3,4)
A = np.array(A)
print(A.shape)

B = [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20]]  # (4,5)
B = np.array(B)
print(B.shape)

# 满秩矩阵
X = [[2, 6, 9],
     [1, 9, 3],
     [7, 2, 4]]

# 矩阵加法
print(np.add(A, A))

# 矩阵乘法
C = np.dot(A, B)
print(C, C.shape)  # (3,4) * (4,5) = (3,5)

# Hadamard积，即对应位置相乘
print(np.multiply(A, A))

# 矩阵转置
A_T = np.transpose(A)
print(A_T, A_T.shape)

# 矩阵的迹
print(np.trace(A))  # 结果为标量

# 行列式
print(np.linalg.det(X))
# print(np.linalg.det(A))  # 报错，计算行列式的矩阵必须为方阵

# 矩阵的秩
print(np.linalg.matrix_rank(X))
print(np.linalg.matrix_rank(A))

# 矩阵的逆
print(np.linalg.inv(X))
# print(np.linalg.inv(A))  # 报错，计算逆的矩阵必须为方阵

# --------------------numpy中的广播机制--------------------

# 通常情况下，numpy两个数组的相加、相减以及相乘都是对应元素之间的操作
x = np.array([[2, 2, 3], [1, 2, 3]])
y = np.array([[1, 1, 3], [2, 2, 4]])
print(x * y)

# 当两个张量维度不同，numpy可以自动使用广播机制使得运算得以完成。例如：
arr = np.random.randn(4, 3)  # (4,3)
arr_mean = arr.mean(axis=0)  # shape(3,)
demeaned = arr - arr_mean  # (4,3) - (3,)
print(demeaned)

# 广播的原则：如果两个数组的后缘维度（trailing dimension，即从末尾开始算起的维度）的轴长度相符，或其中的一方的长度为1，
# 则认为它们是广播兼容的。广播会在缺失和（或）长度为1的维度上进行。

# 广播主要发生在两种情况，一种是两个数组的维数不相等，但是它们的后缘维度的轴长相符，另外一种是有一方的长度为1。

# 1. 数组维度不同，后缘维度的轴长相符
arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])  # (4,3)
arr2 = np.array([1, 2, 3])  # (3,)
arr_sum = arr1 + arr2
print(arr_sum)
# 在上例中，(4,3) + (3,) = (4,3)。类似的例子还有：(3,4,2) + (4,2) = (3,4,2)

# 2. 数组维度相同，其中有个轴为1
arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])  # (4,3)
arr2 = np.array([[1], [2], [3], [4]])  # (4,1)
arr_sum = arr1 + arr2
print(arr_sum)
# 在上例中，(4,3) + (4,1) = (4,3)类似的例子还有：(4,6) + (1,6) = (4,6)；(3,5,6) + (1,5,6) = (3,5,6)；
# (3,5,6) + (3,1,6) = (3,5,6)；(3,5,6) + (3,5,1) = (3,5,6)等
