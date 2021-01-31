# 使用scikit-learn库中的k近邻算法对鸢尾花数据进行分类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def plot_decision_boundary(model, axis):
    """
    在axis范围内绘制模型model的决策边界
    :param model: classification model which must have 'predict' function
    :param axis: [left, right, down, up]
    """
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


X, y = load_iris(return_X_y=True)
X = X[:, :2]  # 仅选择前两个特征，便于绘制决策边界
X_train, X_test, y_train, y_test = train_test_split(X, y)  # 将数据划分为训练集和测试集
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (112, 2) (38, 2) (112,) (38,)

for i, n in enumerate([1, 5, 30]):
    plt.subplot(1, 3, i + 1)

    knn_cls = KNeighborsClassifier(n_neighbors=n)
    knn_cls.fit(X_train, y_train)
    y_pred = knn_cls.predict(X_test)

    print(classification_report(y_test, y_pred))  # 分类报告中包含precision/recall/f1-score

    plt.title('n_neighbors=' + str(n))
    plot_decision_boundary(knn_cls, axis=[3, 8, 1, 5])
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
    plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1])

plt.show()
