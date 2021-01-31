# 使用tensorflow创建卷积神经网络，并对Fashion MNIST衣物图像进行分类

import tensorflow as tf

print(tf.__version__)

# 下载Fashion MNIST数据并进行归一化
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = tf.expand_dims(X_train, -1)  # 扩展通道这一维度
X_test = tf.expand_dims(X_test, -1)  # 扩展通道这一维度

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 将数据转化为tf.data.Dataset类型
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)


class ConvNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
        self.pool1 = tf.keras.layers.AveragePooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.AveragePooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x_conv1 = self.conv1(x)
        x_pool1 = self.pool1(x_conv1)
        x_conv2 = self.conv2(x_pool1)
        x_pool2 = self.pool2(x_conv2)
        x_flatten = self.flatten(x_pool2)
        x_dense1 = self.dense1(x_flatten)
        return self.dense2(x_dense1)


cnn_model = ConvNeuralNetwork()

# 定义损失函数和优化器
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 评价指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# 定义训练步
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = cnn_model(images)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, cnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cnn_model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# 定义测试步
@tf.function
def test_step(images, labels):
    predictions = cnn_model(images)
    loss = loss_function(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


EPOCHS = 10  # 训练轮次

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100,
                          test_loss.result(), test_accuracy.result() * 100))
