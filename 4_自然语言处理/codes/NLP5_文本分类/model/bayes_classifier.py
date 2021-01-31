import numpy as np
import jieba
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.data_loader import proc_original


# 使用多项式朴素贝叶斯算法完成历史试题大类多分类问题

def read_data():
    for file in os.listdir("../data/百度题库/高中_历史/origin"):
        print(file)

    # 读取数据
    df_gd = pd.read_csv(open("../data/百度题库/高中_历史/origin/古代史.csv", encoding='utf8'))
    df_jd = pd.read_csv(open("../data/百度题库/高中_历史/origin/近代史.csv", encoding='utf8'))
    df_xd = pd.read_csv(open("../data/百度题库/高中_历史/origin/现代史.csv", encoding='utf8'))

    # 去掉空数据
    df_gd.dropna()
    df_jd.dropna()
    df_xd.dropna()

    print('古代史数据量: ', len(df_gd))
    print('近代史数据量: ', len(df_jd))
    print('现代史数据量: ', len(df_xd))

    return df_gd, df_jd, df_xd


# 加载停用词
stop_words = pd.read_csv('../data/stopwords/stopwords.txt', encoding='utf-8', sep='\t', header=None, quoting=3)


def process_data(df_data, result, category):
    df_data = proc_original(df_data)
    for row in df_data['item']:
        result.append(row + '|' + category)


def run():
    print('使用朴素贝叶斯算法进行百度题库高中历史题目的三分类')

    df_gd, df_jd, df_xd = read_data()

    data = []
    process_data(df_gd, data, 'gd')
    process_data(df_jd, data, 'jd')
    process_data(df_xd, data, 'xd')

    samples = []
    labels = []
    for item in data:
        sample, label = item.split('|')
        samples.append(sample)
        if label == 'gd':
            labels.append(0)
        elif label == 'jd':
            labels.append(1)
        else:
            labels.append(2)

    labels = np.array(labels)

    # 使用原始词袋模型和tf-idf向量分别训练朴素贝叶斯模型进行分类
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    count_vector = count_vectorizer.fit_transform(samples)
    tfidf_vector = tfidf_vectorizer.fit_transform(samples)

    print(count_vector.shape, tfidf_vector.shape)

    X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(count_vector, labels)
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_vector, labels)

    print(X_train_count.shape, X_test_count.shape, y_train_count.shape, y_test_count.shape)
    print(X_train_tfidf.shape, X_test_tfidf.shape, y_train_tfidf.shape, y_test_tfidf.shape)

    nb_count = MultinomialNB()
    nb_tfidf = MultinomialNB()

    nb_count.fit(X_train_count, y_train_count)
    nb_tfidf.fit(X_train_tfidf, y_train_tfidf)

    y_pred_count = nb_count.predict(X_test_count)
    y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

    print('CountVectorizer提取特征的多项式朴素贝叶斯算法实验结果: ')
    print(classification_report(y_test_count, y_pred_count))
    print('TfidfVectorizer提取特征的多项式朴素贝叶斯算法实验结果:  ')
    print(classification_report(y_test_tfidf, y_pred_tfidf))
