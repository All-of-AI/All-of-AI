# 使用scikit-learn库中的主成分分析算法对Olivetti人脸数据集进行降维及特征脸的提取

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

dataset = fetch_olivetti_faces(shuffle=True)
data = dataset['data']
print(data.shape)  # 原始数据的形状

pca = PCA(n_components=128)
pca.fit(data)

plt.subplot(1, 2, 1)
plt.title('raw first figure')
plt.imshow(data[0].reshape((64, 64)))

data_pca = pca.transform(data)
print(data_pca.shape)  # 降维后数据的形状

data_recover = pca.inverse_transform(data_pca).reshape(400, 64, 64)
print(data_recover.shape)

plt.subplot(1, 2, 2)
plt.title('recovered first figure (n_components=128)')
plt.imshow(data_recover[0])

plt.show()

# 获取主成分，即特征脸
comp = pca.components_
print(comp.shape)

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(comp[i].reshape((64, 64)))
plt.show()
