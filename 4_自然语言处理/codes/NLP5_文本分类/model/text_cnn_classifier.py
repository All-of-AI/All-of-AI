import logging
import argparse
import os
import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from utils.metrics import micro_f1, macro_f1
from utils.data_loader import load_data


def TextCNN(max_sequence_length, max_token_num, embedding_dim, output_dim, model_img_path=None, embedding_matrix=None):
    """
    TextCNN模型
    1. embedding layers
    2. convolution layer
    3. max-pooling
    4. softmax layer
    """
    x_input = Input(shape=(max_sequence_length,))
    if embedding_matrix is None:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length)(x_input)
    else:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                          weights=[embedding_matrix], trainable=True)(x_input)

    # x_embed shape: (None, 300, 500)
    pool_output = []
    kernel_sizes = [2, 3, 4]
    for kernel_size in kernel_sizes:
        # Conv1D input shape: 形如(samples, steps, input_dim)的3D张量
        # Conv1D output shape: 形如(samples，new_steps，nb_filter)的3D张量
        c = Conv1D(filters=100, kernel_size=kernel_size, strides=1)(x_emb)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)

    pool_output = concatenate([p for p in pool_output])  # 将多个池化结果进行拼接
    x_flatten = Flatten()(pool_output)
    y = Dense(output_dim, activation='sigmoid')(x_flatten)  # 95个sigmoid，对应95个类别是否为真(多标签分类的一种方式)

    model = Model([x_input], outputs=[y])
    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    return model


def train(X_train, X_test, y_train, y_test, params, save_path):
    model = build_model(params)
    early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')
    history = model.fit(X_train, y_train, batch_size=params.batch_size, epochs=params.epochs, workers=params.workers,
                        use_multiprocessing=True, callbacks=[early_stopping], validation_data=(X_test, y_test))
    save_model(model, save_path)


def build_model(params):
    if params.model == 'cnn':
        model = TextCNN(max_sequence_length=params.padding_size, max_token_num=params.vocab_size,
                        embedding_dim=params.embed_size, output_dim=params.num_classes)
        model.compile(tf.optimizers.Adam(learning_rate=params.learning_rate), loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])
    else:
        pass
    model.summary()
    return model


def exam(model, x_test, y_true):
    """
    在测试集上验证微平均和宏平均，命名为test会产生程序入口混乱
    """
    y_pred = model.predict(x_test)
    y_true = tf.constant(y_true, tf.float32)
    y_pred = tf.constant(y_pred, tf.float32)
    print('微平均: ', micro_f1(y_true, y_pred).numpy())
    print('宏平均: ', macro_f1(y_true, y_pred).numpy())


def run(MODE='TRAIN'):
    print('使用RNN进行百度题库多标签分类')

    # 设置模型参数
    parser = argparse.ArgumentParser(description='This is the TextCNN train and test project.')
    parser.add_argument('--model', default='cnn')
    parser.add_argument('-t', '--test_sample_percentage', default=0.1, type=float, help='The fraction of test data.')
    parser.add_argument('-p', '--padding_size', default=300, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-e', '--embed_size', default=500, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=100, type=int, help='Number of each convolution kernel.')
    parser.add_argument('-d', '--dropout_rate', default=0.5, type=float, help='Dropout rate in softmax layer.')
    parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float, help='L2 regulation parameter.')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.05, type=float, help='The fraction of validation.')
    parser.add_argument('--results_dir', default='../results/', type=str, help='The results directory.')
    parser.add_argument('--data_path', default='../data/baidu_95.csv', type=str, help='data path')
    parser.add_argument('--vocab_save_dir', default='../data/', type=str, help='data path')
    parser.add_argument('--workers', default=32, type=int, help='use worker count')
    params = parser.parse_args()
    print('Parameters:', params)

    if MODE == 'TRAIN':
        X_train, X_test, y_train, y_test = load_data(params)  # 训练集18060，测试集4516
        train(X_train, X_test, y_train, y_test, params,
              os.path.join(params.results_dir, 'text_cnn_classifier.h5'))  # 训练并保存模型

    elif MODE == 'TEST':
        X_train, X_test, y_train, y_test = load_data(params)
        model = load_model(os.path.join(params.results_dir, 'text_cnn_classifier.h5'),
                           custom_objects={'micro_f1': micro_f1, 'macro_f1': macro_f1})
        exam(model, X_test, y_test)
