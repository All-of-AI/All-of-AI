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
    FILE_NAME = '../data/sentiment_analysis.csv'
    sentence = []
    sentiment = []
    with open(FILE_NAME, 'r', encoding='UTF-8') as csvfile:
        csv_reader = csv.reader(csvfile)  # use csv.reader to read data in csv file
        for row in csv_reader:
            sentence.append(row)
    print('the first sentence is : ', sentence[0])
    for i in range(len(sentence)):
        s = ''
        for item in sentence[i]:
            s += item
        sentence[i] = s.split('\t')[1]
        sentiment.append(s.split('\t')[0])
    print('there are ', len(sentence), ' sentences in total')  # string list
    print('there are ', len(sentiment), 'sentimental labels in total (0/1)')  # 0/1 list

    return sentence, sentiment


# model configuration
VOCAB_SIZE = 5000  # top 5000 frequency words
EMBED_SIZE = 200  # embedding size
NUM_HIDDEN = 128  # hidden size in RNN
BATCH_SIZE = 64
NUM_EPOCH = 5


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

    print('vocabulary size: ', vocab_size)  # 5001, '0' for skipping in embedding layer

    return Xtrain, Xtest, Ytrain, Ytest, vocab_size


def run(MODE='TRAIN'):
    print('使用RNN进行文本情感分析')

    sentence, sentiment = read_data()  # load data
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
        model_rnn.save('../results/rnn_classifier.h5')  # save

    elif MODE == 'TEST':
        model_rnn = load_model('../results/rnn_classifier.h5')
        result = model_rnn.evaluate(Xtest, Ytest)
        print('RNN sentiment analysis accuracy: ', result[1])
