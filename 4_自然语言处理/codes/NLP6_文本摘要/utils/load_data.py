import re
import jieba
import pandas as pd
import numpy as np

from gensim.models.word2vec import LineSentence, Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.params_utils import get_params
from utils.config import *
from utils.multi_proc_utils import parallelize

# 载入词向量参数
params = get_params()
# jieba载入自定义切词表
jieba.load_userdict(user_dict_path)


def build_dataset(train_raw_data_path, test_raw_data_path):
    """
    该函数用于数据加载+预处理(只需执行一次)
    :param train_raw_data_path: 原始训练集路径
    :param test_raw_data_path: 原始测试集路径
    :return: 将中间结果和最终结果保存在data\目录下
    """
    # 1. 加载原始csv数据
    print('1. 加载原始csv数据')
    print(train_raw_data_path)
    train_df = pd.read_csv(train_raw_data_path, engine='python', encoding='utf-8')  # 必须utf-8
    test_df = pd.read_csv(test_raw_data_path, engine='python', encoding='utf-8')  # 必须utf-8
    print('原始csv读出的表格数据类型为: ', type(train_df))  # <class 'pandas.core.frame.DataFrame'>
    print('第一行Question列: ', train_df['Question'][0])
    print('原始训练集行数 {}, 测试集行数 {}'.format(len(train_df), len(test_df)))  # 82943, 20000
    print('\n')

    # 2. 空值去除(对于一行数据，只要该行任意列有空值就去掉该行)
    print('2. 空值去除(对于一行数据，只要该行任意列有空值就去掉该行)')
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    print('空值去除后训练集行数 {}, 测试集行数 {}'.format(len(train_df), len(test_df)))  # 82871, 20000
    print('\n')

    # 3. 多线程, 批量数据预处理(对每个句子执行sentence_proc，清除无用词，切词，过滤停用词，再用空格拼接为一个字符串)
    print('3. 多线程, 批量数据预处理(对每个句子执行sentence_proc，清除无用词，切词，过滤停用词，再用空格拼接为一个字符串)')
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    print('\n')

    # 4. 保存分割处理好的train_seg_data.csv、test_seg_data.csv
    print('4. 保存分割处理好的train_seg_data.csv、test_seg_data.csv')
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)
    print('\n')

    # 5. 将Question和Dialogue用空格连接作为模型输入形成train_df['X']
    print("5. 将Question和Dialogue用空格连接作为模型输入形成train_df['X']")
    train_df['X'] = train_df[['Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    print('\n')

    # 6. 将数据从DataFrame中取出，转为list格式
    print("6. 将数据从DataFrame中取出，转为list格式")
    train_X = train_df['X'].tolist()
    train_y = train_df['Report'].tolist()
    test_X = test_df['X'].tolist()
    print(train_X[:5])
    print('\n')

    # 7. 将texts转为sequences，填充，转变为numpy格式，并保存字典
    print("7. 将texts转为sequences，填充，转变为numpy格式，并保存字典")
    tokenizer = Tokenizer(params['vocab_size'], oov_token='<UNK>')  # 定义tokenizer

    tokenizer.fit_on_texts(train_X)
    tokenizer.fit_on_texts(train_y)
    tokenizer.fit_on_texts(test_X)

    train_X_seq = tokenizer.texts_to_sequences(train_X)
    train_y_seq = tokenizer.texts_to_sequences(train_y)
    test_X_seq = tokenizer.texts_to_sequences(test_X)

    train_X_seq = pad_sequences(train_X_seq, params['max_enc_len'], padding='post')
    train_y_seq = pad_sequences(train_y_seq, params['max_dec_len'], padding='post')
    test_X_seq = pad_sequences(test_X_seq, params['max_enc_len'], padding='post')

    vocab = tokenizer.word_index
    inverse_vocab = tokenizer.index_word
    print('vocab: ', vocab)
    print('inverse_vocab: ', inverse_vocab)
    save_vocab_as_txt('../data/vocab.txt', vocab)
    save_vocab_as_txt('../data/inverse_vocab.txt', inverse_vocab)
    print('\n')

    # 8. 保存numpy数据
    print('8. 保存numpy数据')
    np.save(train_x_path, np.array(train_X_seq))
    np.save(train_y_path, np.array(train_y_seq))
    np.save(test_x_path, np.array(test_X_seq))
    print('\n')

    # 9. 将切分后的数据进一步分为X和Y两部分并保存，便于PGN处理
    print('9. 将切分后的数据进一步分为X和Y两部分并保存，便于PGN处理')
    save_seg_x_y_data()
    print('\n')

    print('数据集构造完毕，于data/目录下')


def save_vocab_as_txt(filename, vocab):
    """
    保存字典到文本文件
    :param filename: 目标txt文件路径
    :param vocab: 要保存的字典
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for k, v in vocab.items():
            f.write("{}\t{}\n".format(k, v))


def load_vocab_from_txt(path):
    """
    从文本文件中读取字典
    :return: 正向字典和反向字典
    """
    vocab = {}
    reverse_vocab = {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f.readlines():
            [word, index] = line.strip("\n").split("\t")
            vocab[word] = int(index)
            reverse_vocab[int(index)] = word
    return vocab, reverse_vocab


def load_train_dataset(max_enc_len=200, max_dec_len=50):
    """
    加载处理好的训练样本和训练标签.npy文件(执行完build_dataset后才能使用)
    :param max_enc_len: 最长样本长度，后面的截断
    :param max_dec_len: 最长标签长度，后面的截断
    :return: 训练样本, 训练标签
    """
    train_X = np.load(train_x_path)
    train_Y = np.load(train_y_path)

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    return train_X, train_Y


def load_test_dataset(max_enc_len=200):
    """
    加载处理好的测试样本.npy文件(执行完build_dataset后才能使用)
    :param max_enc_len: 最长样本长度，后面的截断
    :return: 测试样本
    """
    test_X = np.load(test_x_path)
    test_X = test_X[:, :max_enc_len]
    return test_X


def load_stop_words(stop_word_path):
    """
    加载停用词(程序调用)
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


# 加载停用词
stop_words = load_stop_words(stop_word_path)


def clean_sentence(sentence):
    """
    句子预处理
    :param sentence: 待处理的字符串
    :return: 预处理后的字符串
    """
    # # 1. 将sentence按照'|'分句，并只提取技师的话
    # sub_jishi = []
    # sub = sentence.split('|')  # 按照'|'字符将车主和用户的对话分离
    #
    # for i in range(len(sub)):  # 遍历每个子句
    #     if sub[i].startswith('技师'):  # 只使用技师说的句子
    #         sub_jishi.append(sub[i])
    #
    # sentence = ''.join(sub_jishi)

    # 2. 删除1. 2. 3. 这些标题
    r = re.compile("\D(\d\.)\D")
    sentence = r.sub("", sentence)

    # 3. 删除带括号的 进口 海外
    r = re.compile(r"[(（]进口[)）]|\(海外\)")
    sentence = r.sub("", sentence)

    # 4. 删除除了汉字数字字母和，！？。.- 以外的字符
    r = re.compile("[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")
    # 半角变为全角
    sentence = sentence.replace(",", "，")
    sentence = sentence.replace("!", "！")
    sentence = sentence.replace("?", "？")
    # 问号叹号变为句号
    sentence = sentence.replace("？", "。")
    sentence = sentence.replace("！", "。")
    sentence = r.sub("", sentence)

    # 5. 删除一些无关紧要的词以及语气助词
    r = re.compile(r"车主说|技师说|语音|图片|呢|吧|哈|啊|啦|呕|嗯|吗|不客气")
    sentence = r.sub("", sentence)

    # 6. 删除句子开头的逗号
    if sentence.startswith('，'):
        sentence = sentence[1:]

    return sentence


def filter_stopwords(seg_list):
    """
    过滤一句切好词的话中的停用词(被sentence_proc调用)
    :param seg_list: 切好词的列表 [word1 ,word2 ...]
    :return: 过滤后的停用词
    """
    # 首先去掉多余空字符
    words = [word for word in seg_list if word]
    # 去掉停用词
    return [word for word in words if word not in stop_words]


def sentence_proc(sentence):
    """
    预处理模块(处理一条句子，被sentences_proc调用)
    :param sentence:待处理字符串
    :return: 处理后的字符串
    """
    # 1. 句子清洗
    sentence = clean_sentence(sentence)
    # 2. 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 3. 过滤停用词
    # words = filter_stopwords(words)
    # 4. 拼接成一个字符串,按空格分隔
    return '<START> ' + ' '.join(words) + ' <END>'  # 添加开始符和结束符


def sentences_proc(df):
    """
    预处理模块(处理一个句子列表，对每个句子调用sentence_proc操作)
    :param df: 数据集
    :return: 处理好的数据集
    """
    # 批量预处理 训练集和测试集
    for col_name in ['Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


def save_seg_x_y_data():
    """
    将seg的train和test细分为seg_X和seg_Y，便于PGN的处理
    """
    # 读取数据
    train_seg_df = pd.read_csv(train_seg_path, engine='python', encoding='utf-8').fillna("")
    test_seg_df = pd.read_csv(test_seg_path, engine='python', encoding='utf-8').fillna("")

    # 删除Report为空的样本
    na_idx = train_seg_df[(train_seg_df["Report"] == "") | (train_seg_df["Report"] == " ")].index
    train_seg_df.drop(na_idx, axis=0, inplace=True)

    # 构建训练集的X和Y以及测试集的X
    train_seg_df['train_seg_X'] = train_seg_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_seg_df['train_seg_Y'] = train_seg_df['Report']
    test_seg_df['test_seg_X'] = test_seg_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 保存切割后的X和Y数据为.csv文件，记得header=False
    train_seg_df['train_seg_X'].to_csv(train_x_seg_path, index=None, header=False)
    train_seg_df['train_seg_Y'].to_csv(train_y_seg_path, index=None, header=False)
    test_seg_df['test_seg_X'].to_csv(test_x_seg_path, index=None, header=False)


if __name__ == '__main__':
    # 数据集批量处理，只需执行一次
    build_dataset(train_raw_data_path, test_raw_data_path)
    train_X, train_y = load_train_dataset()
    test_X = load_test_dataset()
    print(train_X.shape, train_y.shape, test_X.shape)  # (82871, 200) (82871, 50) (20000, 200)
