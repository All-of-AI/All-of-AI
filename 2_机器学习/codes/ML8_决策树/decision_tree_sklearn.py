# 使用scikit-learn库中的决策树算法对鸢尾花数据进行分类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (112, 2) (38, 2) (112,) (38,)

# 尝试不同限定深度的决策树，并绘制其决策边界
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

dec_tree_10 = DecisionTreeClassifier(max_depth=10)
dec_tree_10.fit(X_train, y_train)

dec_tree_6 = DecisionTreeClassifier(max_depth=6)
dec_tree_6.fit(X_train, y_train)

dec_tree_4 = DecisionTreeClassifier(max_depth=4)
dec_tree_4.fit(X_train, y_train)

plt.subplot(2, 2, 1)
plt.title('No max depth')
plot_decision_boundary(dec_tree, axis=[3, 8, 1, 5])
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1])

plt.subplot(2, 2, 2)
plt.title('Max depth = 10')
plot_decision_boundary(dec_tree_10, axis=[3, 8, 1, 5])
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1])

plt.subplot(2, 2, 3)
plt.title('Max depth = 6')
plot_decision_boundary(dec_tree_6, axis=[3, 8, 1, 5])
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1])

plt.subplot(2, 2, 4)
plt.title('Max depth = 4')
plot_decision_boundary(dec_tree_4, axis=[3, 8, 1, 5])
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1])
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1])
plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1])

print('classification accuracy when max depth is with no limitation: ', dec_tree.score(X_test, y_test))
print('classification accuracy when max depth is 10: ', dec_tree_10.score(X_test, y_test))
print('classification accuracy when max depth is 6: ', dec_tree_6.score(X_test, y_test))
print('classification accuracy when max depth is 4: ', dec_tree_4.score(X_test, y_test))

plt.show()
