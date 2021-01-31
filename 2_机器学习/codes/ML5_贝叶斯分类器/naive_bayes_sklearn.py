# 使用scikit-learn库中的多项式朴素贝叶斯算法对新闻文本进行分类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 加载数据，vectorized数据表示已将文本变为向量表示(这里使用tf-idf作为特征)
newsgroups_train = fetch_20newsgroups_vectorized('train')
X_train = newsgroups_train['data']
y_train = newsgroups_train['target']

newsgroups_test = fetch_20newsgroups_vectorized('test')
X_test = newsgroups_test['data']
y_test = newsgroups_test['target']

y_train = np.array(y_train)  # 变为numpy数组
y_test = np.array(y_test)  # 变为numpy数组

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (11314, 130107) (7532, 130107) (11314,) (7532,)

nb = MultinomialNB()  # 定义多项式朴素贝叶斯分类器

nb.fit(X_train, y_train)

print(accuracy_score(y_test, nb.predict(X_test)))  # 打印分类准确率
print(classification_report(y_test, nb.predict(X_test)))  # 分类报告中包含precision/recall/f1-score
