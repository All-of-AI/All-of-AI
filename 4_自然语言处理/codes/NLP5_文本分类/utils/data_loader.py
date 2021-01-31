import argparse
import os
import numpy as np
import jieba
import re
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.config import root
from utils.multi_proc_utils import parallelize


def load_stop_words(stop_word_path):
    """
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    """
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


def clean_sentence(line):
    """
    清除无用词+切词
    :param line: 输入的句子
    :return: 清除无用词后并分词的列表
    """
    line = re.sub(
        "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(\"\')]+|[:：+—()?【】《》“”！，。？、~@#￥%…&*（）]+|题目", '', line)
    words = jieba.cut(line, cut_all=False)
    return words


stopwords_path = '../data/stopwords/哈工大停用词表.txt'
stop_words = load_stop_words(stopwords_path)


def sentence_proc(sentence):
    """
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    """
    # 清除无用词+切词
    words = clean_sentence(sentence)
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def proc_original(df):
    """
    批量处理原始数据集
    """
    df['item'] = df['item'].apply(sentence_proc)
    return df


def proc(df):
    """
    批量处理读取后的数据集
    """
    df['content'] = df['content'].apply(sentence_proc)
    return df


def load_data(params, is_rebuild_dataset=False):
    if os.path.exists(os.path.join(root, 'data', 'X_train.npy')) and not is_rebuild_dataset:
        X_train = np.load(os.path.join(root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(root, 'data', 'y_test.npy'))
        return X_train, X_test, y_train, y_test

    # 读取数据
    df = pd.read_csv(params.data_path, header=None).rename(columns={0: 'label', 1: 'content'})
    # 并行清理数据
    df = parallelize(df, proc)
    # word2index
    text_preprocesser = Tokenizer(num_words=params.vocab_size, oov_token="<UNK>")
    text_preprocesser.fit_on_texts(df['content'])
    # save vocab
    word_dict = text_preprocesser.word_index
    with open(params.vocab_save_dir + 'vocab.txt', 'w', encoding='utf-8') as f:
        for k, v in word_dict.items():
            f.write(f'{k}\t{str(v)}\n')

    x = text_preprocesser.texts_to_sequences(df['content'])
    # 填充
    x = pad_sequences(x, maxlen=params.padding_size, padding='post', truncating='post')
    # 划分标签
    df['label'] = df['label'].apply(lambda x: x.split())
    # 多标签编码
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['label'])
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 保存数据
    np.save(os.path.join(root, 'data', 'X_train.npy'), X_train)
    np.save(os.path.join(root, 'data', 'X_test.npy'), X_test)
    np.save(os.path.join(root, 'data', 'y_train.npy'), y_train)
    np.save(os.path.join(root, 'data', 'y_test.npy'), y_test)

    return X_train, X_test, y_train, y_test


def data2file(file_dir):
    """
    4层标签(带知识点)数据处理，仅需调用一次
    生成文件: {file_dir}/baidu_{label_num}.csv
    :param file_dir: 原始数据集文件路径
    """
    grades = ['高中']
    subjects = ['地理', '历史', '生物', '政治']
    categories = {'地理': ['地球与地图', '宇宙中的地球', '生产活动与地域联系', '人口与城市', '区域可持续发展'],
                  '历史': ['古代史', '近代史', '现代史'],
                  '生物': ['现代生物技术专题', '生物科学与社会', '生物技术实践', '稳态与环境', '遗传与进化', '分子与细胞'],
                  '政治': ['经济学常识', '科学思维常识', '生活中的法律常识', '科学社会主义常识', '公民道德与伦理常识', '时事政治']
                  }

    df_target = pd.DataFrame(columns=['labels', 'item'])
    for grade in grades:
        for subject in subjects:
            for category in categories[subject]:
                # data/百度题库/高中_历史/origin/古代史.csv
                file = os.path.join(file_dir, grade + '_' + subject, 'origin', category + '.csv')
                df = pd.read_csv(open(file, encoding='utf8'))  # 先open，后read_csv的读取方式
                print(f'{grade} {subject} {category} \tsize:{len(df)}')

                # 按网页顺序对其排序
                df['web-scraper-order'] = df['web-scraper-order'].apply(lambda x: int(x.split('-')[1]))
                df = df[['web-scraper-order', 'item']]
                df = df.sort_values(by='web-scraper-order')

                # 删除文本中的换行
                df['item'] = df.item.apply(lambda x: "".join(x.split()))
                df['labels'] = df.item.apply(
                    lambda x: [grade, subject, category] + x[x.index('[知识点：]') + 6:].split(',') if x.find(
                        '[知识点：]') != -1 else [grade, subject, category])
                df['item'] = df.item.apply(lambda x: x.replace('[题目]', ''))
                df['item'] = df.item.apply(lambda x: x[:x.index('题型')] if x.index('题型') else x)

                df = df[['labels', 'item']]
                df_target = df_target.append(df)

    print('origin data size:', len(df_target))
    print(df_target.head())

    # 设置样本数量阈值
    min_samples = 300
    # 阈值 标签数
    # 500  64
    # 400  75
    # 300  95
    # 200  134
    # 100  228

    df = df_target.copy()
    labels = []
    for i in df.labels:
        labels.extend(i)

    result = dict(sorted(dict(Counter(labels)).items(), key=lambda x: x[1], reverse=True))
    lens = np.array(list(result.values()))
    LABEL_NUM = len(lens[lens > min_samples])

    # 选定数据label
    label_target = set([k for k, v in result.items() if v > min_samples])

    # 保证grade subject category在前三位置
    df['labels'] = df.labels.apply(lambda x: x[:3] + list(set(x) - set(x[:3]) & label_target))
    df['labels'] = df.labels.apply(lambda x: None if len(x) < 4 else x)  # 去除没有知识点的数据
    df = df[df.labels.notna()]

    # 最终的label数量
    labels = []
    [labels.extend(i) for i in df.labels]
    LABEL_NUM = len(set(labels))

    print(f'>{min_samples} datasize:{len(df)} multi_class:{LABEL_NUM}')

    profix = ''
    if profix:
        df['labels'] = df.labels.apply(lambda x: [profix + i for i in x])
    df['labels'] = df.labels.apply(lambda x: ' '.join(x))
    # shuffle
    df = df.sample(frac=1)
    file = os.path.join(file_dir, f'baidu_{LABEL_NUM}{profix}.csv')
    df.to_csv(file, index=False, sep=',', header=False, encoding='UTF8')  # 当sep字符在df中存在会在字符串前后添加引号
    print('csv data file generated! ', file)


def data2file_without_knowledge(file_dir):
    """
    三层标签数据生成，不包含详细知识点，仅需调用一次
    :param file_dir: 原始数据集文件路径
    """
    grades = ['高中']
    subjects = ['地理', '历史', '生物', '政治']
    categories = {'地理': ['地球与地图', '宇宙中的地球', '生产活动与地域联系', '人口与城市', '区域可持续发展'],
                  '历史': ['古代史', '近代史', '现代史'],
                  '生物': ['现代生物技术专题', '生物科学与社会', '生物技术实践', '稳态与环境', '遗传与进化', '分子与细胞'],
                  '政治': ['经济学常识', '科学思维常识', '生活中的法律常识', '科学社会主义常识', '公民道德与伦理常识', '时事政治']
                  }

    for grade in grades:
        for subject in subjects:
            for category in categories[subject]:
                file = os.path.join(file_dir, f'{grade}_{subject}', 'origin', category + '.csv')
                df = pd.read_csv(open(file, encoding='utf8'))
                print('size:', len(df))

                # 按网页顺序对其排序
                df['web-scraper-order'] = df['web-scraper-order'].apply(lambda x: int(x.split('-')[1]))
                df = df[['web-scraper-order', 'item']]
                df = df.sort_values(by='web-scraper-order')

                # 对文本处理
                def foo(x):
                    x = ''.join(x.split())
                    x = x[:x.index('题型')]
                    return x

                # 删除文本中的换行
                df['item'] = df.item.apply(lambda x: foo(x))
                df = df['item']

                with open(os.path.join(file_dir, f'{grade}_{subject}', category + '_test.csv'), 'w',
                          encoding='utf8') as f:
                    f.write('\n'.join(list(df.values)))
                print(category + 'Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('-d', '--data_path', default='../data/baidu_95.csv', type=str, help='data path of baidu_95.csv')
    parser.add_argument('-v', '--vocab_save_dir', default='../data/', type=str, help='data path')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('-p', '--padding_size', default=300, type=int, help='Padding size of sentences.(default=300)')

    params = parser.parse_args()
    print('Parameters:', params)

    # X_train, X_test是训练集、测试集的语料序列，每条序列长度为300，后部补0
    # y_train, y_test是训练集、测试集的类别标签，共95个类别
    X_train, X_test, y_train, y_test = load_data(params)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)  # (18060, 300) (4516, 300) (18060, 95) (4516, 95)
    print('第一条训练语料序列: ')
    print(X_train[0])
    print('第一条类别标签: ')
    print(y_train[0])
