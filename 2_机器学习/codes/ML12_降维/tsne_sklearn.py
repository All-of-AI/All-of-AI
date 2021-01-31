# 使用scikit-learn库中的t-SNE算法完成恒星光谱数据的降维

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.manifold import TSNE

# 数据获取
# A F K M类恒星数据各1000条
X_a = sio.loadmat('spectra_data\A.mat')['P1'][:1000]
X_f = sio.loadmat('spectra_data\F.mat')['P1'][:1000]
X_k = sio.loadmat('spectra_data\K.mat')['P1'][:1000]
X_m = sio.loadmat('spectra_data\M.mat')['P1'][:1000]
X_label = ['A', 'F', 'K', 'M']
X = np.vstack((X_a, X_f, X_k, X_m))

# 类别标签
y_a = np.full((X_a.shape[0],), 0)
y_f = np.full((X_f.shape[0],), 1)
y_k = np.full((X_k.shape[0],), 2)
y_m = np.full((X_m.shape[0],), 3)
y = np.hstack((y_a, y_f, y_k, y_m))

# 数据归一化
for i in range(X.shape[0]):
    X[i] -= np.min(X[i])
    if np.max(X[i]) != 0:
        X[i] /= np.max(X[i])

print('Data shape: ', X.shape)
print('Label shape: ', y.shape)

# 降维
tsne = TSNE()
X_tsne = tsne.fit_transform(X)

# 绘制降维结果
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='A')
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='F')
plt.scatter(X_tsne[y == 2, 0], X_tsne[y == 2, 1], label='K')
plt.scatter(X_tsne[y == 3, 0], X_tsne[y == 3, 1], label='M')
plt.legend()
plt.show()
