# 使用scikit-learn库中的逻辑回归算法完成自定义数据集上的分类任务

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


# 使用numpy库构建自定义数据集
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array((X[:, 0] ** 2 + X[:, 1]) < 1.5, dtype='int')
for _ in range(20):
    y[np.random.randint(200)] = 1

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()  # 定义逻辑回归类的对象
log_reg.fit(X_train, y_train)  # 训练

print('classification accuracy of original logistic regression: ', log_reg.score(X_test, y_test))  # 评分函数

# 绘制原始逻辑回归模型的决策边界
plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()


# 使用Pipeline添加多项式特征、归一化后再应用逻辑回归算法
def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)

print('classification accuracy of polynomial logistic regression: ', poly_log_reg.score(X_test, y_test))

# 绘制添加了多项式特征后的逻辑回归算法的决策边界
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
