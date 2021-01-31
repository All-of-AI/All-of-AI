# 使用tensroflow构造自编码器，完成恒星光谱数据的降维与重构

import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 数据获取
# A F K M类恒星数据各6000条
X_a = sio.loadmat('spectra_data\A.mat')['P1']
X_f = sio.loadmat('spectra_data\F.mat')['P1']
X_k = sio.loadmat('spectra_data\K.mat')['P1']
X_m = sio.loadmat('spectra_data\M.mat')['P1']
X_label = ['A', 'F', 'K', 'M']
X = np.vstack((X_a, X_f, X_k, X_m))

# 数据归一化
for i in range(X.shape[0]):
    X[i] -= np.min(X[i])
    if np.max(X[i]) != 0:
        X[i] /= np.max(X[i])

print('Data shape: ', X.shape)

# 转为tf.data.Dataset格式
data = tf.data.Dataset.from_tensor_slices(X).shuffle(24000).batch(64, drop_remainder=True)
print(data)


# 定义自编码器模型
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_1 = tf.keras.layers.Dense(64)
        self.encoder_2 = tf.keras.layers.Dense(10)
        self.decoder_1 = tf.keras.layers.Dense(64)
        self.decoder_2 = tf.keras.layers.Dense(3522)

    def call(self, x):
        x = self.encoder_1(x)
        coding = self.encoder_2(x)
        x = self.decoder_1(coding)
        rebuild = self.decoder_2(x)

        return coding, rebuild


ae = AutoEncoder()

# 定义损失函数和优化器
loss_func = tf.losses.MeanSquaredError()
optimizer = tf.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')


@tf.function
def train_step(batch_data):
    with tf.GradientTape() as tape:
        coding, rebuild = ae(batch_data)
        loss = loss_func(batch_data, rebuild)
    gradients = tape.gradient(loss, ae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ae.trainable_variables))

    train_loss(loss)


EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss.reset_states()

    for batch_data in data:
        train_step(batch_data)

    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch + 1, train_loss.result()))

sample = X[:5]
sample_coding, sample_rebuild = ae(sample)
print(sample.shape, sample_coding.shape, sample_rebuild.shape)

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.plot(sample[i])

    plt.subplot(2, 5, i + 5 + 1)
    plt.plot(sample_rebuild[i])

plt.show()
