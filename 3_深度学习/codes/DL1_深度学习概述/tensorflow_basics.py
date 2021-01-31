# tensorflow库的基本操作

import tensorflow as tf
import numpy as np

# 在tensorflow中，tensor是数据的基本元素。tensor可以是标量(0维)、向量(1维)、矩阵(2维)等。例如：

# 定义一个随机数(标量)
random_float = tf.random.uniform(shape=())

# 定义一个含有两个元素的零向量
zero_vector = tf.zeros(shape=2)

# 定义两个(2, 2)大小的矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# tensor的三个最重要的属性是形状(shape)、数据类型(dtype)和值(.numpy())。例如：
print(A.shape)  # (2, 2)
print(A.dtype)  # <dtype: 'float32'>
print(A.numpy())  # [[1. 2.], [3. 4.]]

# 在tensorflow中可以对tensor进行许多运算操作:
C = tf.add(A, B)  # 矩阵加法
D = tf.matmul(A, B)  # 矩阵乘法
print(C)
print(D)

# tensorflow中的单变量自动微分
x = tf.Variable(initial_value=3.)  # 定义一个变量
with tf.GradientTape() as tape:  # 在tf.GradientTape()上下文中，所有的计算都会被记录，用于自动微分的计算
    y = tf.square(x)
y_grad = tape.gradient(y, x)  # 计算y对于x的梯度
print(y, y_grad)

# # tensorflow中的多变量自动微分
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])  # 计算函数L(w, b)对w和b的梯度
print(L, w_grad, b_grad)

# 使用tensorflow实现线性回归
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
    print(a, b)
