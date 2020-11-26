# 利用numpy库计算矩阵的特征值和特征向量，以及对矩阵进行奇异值分解

import numpy as np
import matplotlib.pyplot as plt

# --------------------特征值分解--------------------
X = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

eigenvalues, eigenvectors = np.linalg.eig(X)

print("eigenvalue: \n", eigenvalues)
print("eigenvector: \n", eigenvectors)

# --------------------奇异值分解--------------------
U, Sigma, V = np.linalg.svd(X, )

print("U: \n", U)
print("Sigma: \n", Sigma)
print("V: \n", V)

# 通过截断奇异值分解重建矩阵X
Sigma[1:] = 0
X_rebuild = np.mat(U) * np.mat(np.diag(Sigma)) * np.mat(V)
print(X_rebuild)  # 矩阵X的近似

# --------------------奇异值分解在图像中的应用--------------------
lenna = plt.imread('lenna.jpg')
print(lenna.shape)  # (2318,1084,3)

lenna = lenna[:1000, :1000, 2]  # 将图像大小调整为(1000,1000)，且仅选取一个通道
print(lenna.shape)  # (1000,1000)

U, Sigma, V = np.linalg.svd(lenna)

print(U.shape, Sigma.shape, V.shape)

# 通过截断奇异值分解重建莱娜图，观察k取不同值时的重建结果
k = [1000, 500, 300, 200, 100, 50]
for i in range(len(k)):
    Sigma_k = np.copy(Sigma)
    Sigma_k[k[i]:] = 0
    lenna_rebuild = np.mat(U) * np.mat(np.diag(Sigma_k)) * np.mat(V)

    plt.subplot(2, 3, i + 1)
    plt.title('truncated k={}'.format(k[i]))
    plt.imshow(lenna_rebuild)

plt.show()
