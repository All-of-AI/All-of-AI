from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import numpy as np
import csv

# 使用fastText对句子情感进行0/1分类
np.random.seed(42)


def read_data():
    # 读取数据
    FILE_NAME = '../data/sentiment_analysis.csv'
    sentence = []
    sentiment = []
    with open(FILE_NAME, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv文件中的数据保存到birth_data中
            sentence.append(row)
    print('第一条句子为: ', sentence[0])
    for i in range(len(sentence)):
        s = ''
        for item in sentence[i]:
            s += item
        sentence[i] = s.split('\t')[1]
        sentiment.append(s.split('\t')[0])
    print('共有', len(sentence), '个句子')  # 字符串list
    print('共有', len(sentiment), '个标签，均为0/1')  # 0/1标签list

    return sentence, sentiment


# 模型配置
VOCAB_SIZE = 5000  # 只考虑前5000词频的词汇
EMBED_SIZE = 200  # 嵌入空间大小
NUM_HIDDEN = 128  # 隐藏层神经元个数
BATCH_SIZE = 64
NUM_EPOCH = 5


def process_data(sentence, sentiment):
    # 分词、填充以及获取词典
    tokenizer = Tokenizer(VOCAB_SIZE, oov_token='<UNK>')
    tokenizer.fit_on_texts(sentence)
    sequence = tokenizer.texts_to_sequences(sentence)
    X = pad_sequences(sequence, padding='post')
    X = X[:, :120]  # 只取前120个词
    Y = np.array(sentiment)

    # 对于Tokenizer，设置其词典大小(例如5000)后，其产生的词典还是会将所有文本中的单词收纳进去。但是转化成sequence时，只有词频在词典
    # 大小范围内的单词才会被转化为其对应的index，词频在词典大小之外的单词全部被转化为<UNK>对应的index
    vocab = tokenizer.word_index
    print(vocab)
    print(len(vocab))

    vocab_size = VOCAB_SIZE + 1

    print('X如下: ')
    print(X)
    print("Y如下: ")
    print(Y)

    print('X大小为: ', X.shape)
    print('Y大小为: ', Y.shape)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)

    print('字典大小为: ', vocab_size)  # 5001，因为是0~5000，0作为空字符

    return Xtrain, Xtest, Ytrain, Ytest, vocab_size


def run(MODE='TRAIN'):
    print('使用FastText算法进行文本情感分析')

    sentence, sentiment = read_data()  # 读取数据
    Xtrain, Xtest, Ytrain, Ytest, vocab_size = process_data(sentence, sentiment)

    if MODE == 'TRAIN':
        # 构建模型
        fasttext_model = Sequential()
        fasttext_model.add(Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE, input_length=120, mask_zero=True))
        fasttext_model.add(GlobalAveragePooling1D())
        fasttext_model.add(Dense(NUM_HIDDEN, activation='relu'))
        fasttext_model.add(Dense(2, activation='softmax'))
        fasttext_model.summary()

        fasttext_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = fasttext_model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCH,
                                     validation_data=(Xtest, Ytest))
        fasttext_model.save('../results/fasttext_classifier.h5')  # 保存为h5格式

    elif MODE == 'TEST':
        fasttext_model = load_model('../results/fasttext_classifier.h5')
        result = fasttext_model.evaluate(Xtest, Ytest)
        print('FastText情感分析分类准确率: ', result[1])
