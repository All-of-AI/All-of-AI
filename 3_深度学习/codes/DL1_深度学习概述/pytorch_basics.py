# pytorch库的基本操作

import torch
import numpy as np

# 和tensorflow类似，tensor也是pytorch中的数据基本元素

# 构建一个未初始化的(5, 3)大小的矩阵
x = torch.empty(5, 3)
print(x)

# 创建一个随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个元素全为0的矩阵，数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接使用数据创建tensor
x = torch.tensor([5.5, 3])
print(x)

# 基于现有的tensor创建一个新的tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)  # 重写数据类型
print(x)  # 结果的shape相同

# 获取一个tensor的大小
print(x.size())

# pytorch中tensor的加法操作的不同写法
y = torch.rand(5, 3)

print(x + y)  # 写法1

print(torch.add(x, y))  # 写法2

result = torch.empty(5, 3)
torch.add(x, y, out=result)  # 写法3
print(result)

y.add_(x)  # 写法4
print(y)

# 像numpy一样使用切片和索引访问tensor中的元素
print(x[:, 1])

# 使用torch.view()函数来改变tensor的形状
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# 对于只有一个元素的tensor，可以使用.item()去获取python的数据类型
x = torch.randn(1)
print(x)
print(x.item())

# 将tensor转化为numpy数组
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# 将numpy数组转化为tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# 自动微分

# 建立一个tensor，并设置requires_grad=True来跟踪其计算过程
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 进行一个tensor操作
y = x + 2
print(y)

# y作为运算的结果，其含有属性grad_fn
print(y.grad_fn)

# 在y上进行更多的操作
z = y * y * 3
out = z.mean()
print(z, out)

# 使用.requires_grad_(...)函数改变一个tensor的requires_grad属性，默认为False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 反向传播过程。由于out仅包含一个标量，因此out.backward()等价于out.backward(torch.tensor(1.))
out.backward()
# 打印梯度d(out)/dx
print(x.grad)

# 向量的Jacobian积
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# 现在在这种情况下，y不再是一个标量。torch.autograd不能够直接计算整个雅可比，但是如果我们只想要雅可比向量积，
# 只需要简单的传递向量给backward作为参数
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# 可以通过将代码包裹在with torch.no_grad()，来停止对从跟踪历史中的.requires_grad=True的张量自动求导。

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
