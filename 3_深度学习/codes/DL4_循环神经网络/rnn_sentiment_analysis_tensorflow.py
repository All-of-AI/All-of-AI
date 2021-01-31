# 使用tensorflow创建循环神经网络，并对文本情感进行分析

import numpy as np
import csv
import tensorflow as tf

from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

print(tf.__version__)


def read_data():
    FILE_NAME = 'sentiment_analysis.csv'
    sentence = []
    sentiment = []
    with open(FILE_NAME, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader来读取csv文件
        for row in csv_reader:
            sentence.append(row)
    print('the first sentence is : ', sentence[0])
    for i in range(len(sentence)):
        s = ''
        for item in sentence[i]:
            s += item
        sentence[i] = s.split('\t')[1]
        sentiment.append(s.split('\t')[0])
    print('there are ', len(sentence), ' sentences in total')  # 字符串列表
    print('there are ', len(sentiment), 'sentimental labels in total (0/1)')  # 0/1的列表

    return sentence, sentiment


# 模型配置
VOCAB_SIZE = 5000  # 选取前5000个最频繁出现的单词
EMBED_SIZE = 200  # 词嵌入大小
NUM_HIDDEN = 128  # RNN中隐含层神经元个数
BATCH_SIZE = 64
NUM_EPOCH = 3


def process_data(sentence, sentiment):
    # word segmentation, padding and obtaining vocabulary
    tokenizer = Tokenizer(VOCAB_SIZE, oov_token='<UNK>')  # 1. define tokenizer
    tokenizer.fit_on_texts(sentence)  # 2. fit on texts
    sequence = tokenizer.texts_to_sequences(sentence)  # 3. texts to sequences
    X = pad_sequences(sequence, padding='post')  # 4. pad sequences
    X = X[:, :120]  # length cut
    Y = np.array(sentiment)

    # for Tokenizer, although when the vocabulary size has been set, for example, 5000, the vocabulary will contain all
    # words in texts. However, when transfering texts to sequences, only words with top 5000 frequency will be converted
    # to corresponding index, other words will be converted to <UNK>
    vocab = tokenizer.word_index
    print(vocab)
    print(len(vocab))

    vocab_size = VOCAB_SIZE + 1

    print('X: ')
    print(X)
    print("Y: ")
    print(Y)

    print('X size: ', X.shape)
    print('Y size: ', Y.shape)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)

    print('vocabulary size: ', vocab_size)  # 5001，'0'用于填充

    return Xtrain, Xtest, Ytrain, Ytest, vocab_size


def run(MODE='TRAIN'):
    sentence, sentiment = read_data()  # 加载数据
    Xtrain, Xtest, Ytrain, Ytest, vocab_size = process_data(sentence, sentiment)

    if MODE == 'TRAIN':
        model_rnn = Sequential()
        model_rnn.add(Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE, input_length=120, mask_zero=True))
        model_rnn.add(SimpleRNN(NUM_HIDDEN, activation='relu'))  # LSTM/GRU
        model_rnn.add(Dense(NUM_HIDDEN, activation='relu'))
        model_rnn.add(Dense(2, activation='softmax'))
        model_rnn.summary()

        model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model_rnn.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, validation_data=(Xtest, Ytest))
        model_rnn.save('rnn_classifier.h5')  # 保存模型为h5格式

    elif MODE == 'TEST':
        model_rnn = load_model('rnn_classifier.h5')
        result = model_rnn.evaluate(Xtest, Ytest)
        print('RNN sentiment analysis accuracy: ', result[1])


run('TRAIN')
run('TEST')
