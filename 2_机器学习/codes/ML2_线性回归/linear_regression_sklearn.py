# 使用scikit-learn库中的线性回归算法对波士顿房价进行预测

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

boston_data = load_boston()  # 载入波士顿房价数据

X = boston_data['data']
y = boston_data['target']
feature_names = boston_data['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 将数据划分为训练集和测试集

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # 共506个样本，每个样本有13个特征
print(feature_names)  # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# 线性回归
lin_reg = LinearRegression()  # 定义线性回归对象
lin_reg.fit(X_train, y_train)  # 训练
y_pred = lin_reg.predict(X_test)  # 测试
print('r2 score in test set using 13 original features: ', r2_score(y_test, y_pred))  # R2 score约为67.7

# 为原始数据添加多项式特征
poly_feature = PolynomialFeatures(degree=2)  # 当degree>2时，出现过拟合现象

X_train_poly = poly_feature.fit_transform(X_train)
X_test_poly = poly_feature.fit_transform(X_test)
print(X_train_poly.shape, X_test_poly.shape)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_train_poly, y_train)
y_pred = lin_reg_poly.predict(X_test_poly)
print('r2 score in test set using polynomial features: ', r2_score(y_test, y_pred))  # R2 score约为81.8

# 仅选择一个特征(5)对数据和模型进行可视化
line_reg_one = LinearRegression()
line_reg_one.fit(X_train[:, 5].reshape(-1, 1), y_train)
y_pred = line_reg_one.predict(X_test[:, 5].reshape(-1, 1))

# 绘制数据分布
plt.scatter(X_train[:, 5], y_train, label='training set')
plt.scatter(X_test[:, 5], y_test, label='test set')

# 绘制训练得到的模型
x_plot = np.linspace(3, 9, 100)
y_plot = line_reg_one.predict(x_plot.reshape(-1, 1))
plt.plot(x_plot, y_plot, color='red')
plt.title('linear regression with one feature')
plt.legend()
plt.show()

# 欠拟合、拟合和过拟合在线性回归中的的示例
X_fitting = np.linspace(-2, 2, 100)
y_fitting = X_fitting * X_fitting + np.random.normal(0, 0.5, 100)

lin_reg_fitting = LinearRegression()

poly_list = [1, 2, 50]
for i in range(len(poly_list)):
    poly_feature_fitting = PolynomialFeatures(degree=poly_list[i])
    X_poly_fitting = poly_feature_fitting.fit_transform(X_fitting.reshape(-1, 1))
    print('dimensions of X_fitting after adding polynomial features of {} degree: {}'.format(poly_list[i],
                                                                                             X_poly_fitting.shape))

    lin_reg_fitting.fit(X_poly_fitting, y_fitting)
    y_predict_fitting = lin_reg_fitting.predict(poly_feature_fitting.fit_transform(X_fitting.reshape(-1, 1)))
    plt.plot(X_fitting, y_predict_fitting, label='degree: {}'.format(poly_list[i]))

plt.scatter(X_fitting, y_fitting)
plt.title('under-fitting, fitting and over-fitting')
plt.legend()
plt.show()
