# 比较不同的优化器的优化效果

import tensorflow as tf
import matplotlib.pyplot as plt

# 下载MNIST数据并进行归一化
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 定义网络结构
def define_network():
    neural_network = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return neural_network


# 定义多个网络，后期使用不同的优化器进行优化
nn_sgd = define_network()
nn_adagrad = define_network()
nn_adam = define_network()
nn_rms = define_network()
nn_delta = define_network()

# 配置并训练优化方式不同的模型
# SGD
print('SGD')
nn_sgd.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_sgd = nn_sgd.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test), verbose=2)

# AdaGrad
print('AdaGrad')
nn_adagrad.compile(optimizer=tf.keras.optimizers.Adagrad(), loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
history_adagrad = nn_adagrad.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test),
                                 verbose=2)

# Adam
print('Adam')
nn_adam.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_adam = nn_adam.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test), verbose=2)

# RMSprop
print('RMSprop')
nn_rms.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_rms = nn_rms.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test), verbose=2)

# AdaDelta
print('AdaDelta')
nn_delta.compile(optimizer=tf.keras.optimizers.Adadelta(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_delta = nn_delta.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test), verbose=2)

# 绘制不同优化器的优化过程
# loss
plt.subplot(1, 2, 1)
plt.plot(history_sgd.history['loss'], label='SGD')
plt.plot(history_adagrad.history['loss'], label='AdaGrad')
plt.plot(history_adam.history['loss'], label='Adam')
plt.plot(history_rms.history['loss'], label='RMSprop')
plt.plot(history_delta.history['loss'], label='AdaDelta')
plt.legend()

# accuracy
plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['accuracy'], label='SGD')
plt.plot(history_adagrad.history['accuracy'], label='Adagrad')
plt.plot(history_adam.history['accuracy'], label='Adam')
plt.plot(history_rms.history['accuracy'], label='RMSprop')
plt.plot(history_delta.history['accuracy'], label='AdaDelta')
plt.legend()

plt.show()
