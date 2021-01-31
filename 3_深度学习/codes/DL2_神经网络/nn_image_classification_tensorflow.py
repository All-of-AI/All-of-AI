# 使用tensorflow中的两种方式创建神经网络，并对MNIST手写数字进行识别

import tensorflow as tf

print(tf.__version__)

# 下载MNIST数据并进行归一化
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 第一种构造模型的方式：使用keras序列化API，该方法仅能构造线性神经网络结构
print('1. use keras sequential API to build a neural network')
neural_network_beginner = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 打印模型信息
neural_network_beginner.summary()

# complie函数中的常用配置：优化器、损失函数和评价指标
# categorical_crossentropy和sparse_categorical_crossentropy都是计算多分类crossentropy的，只是对y的格式要求不同。
# 如果是categorical_crossentropy，那y必须为one-hot形式
# 如果是sparse_categorical_crossentropy，那y就是原始的整数形式，比如[1, 0, 2, 0, 2]
neural_network_beginner.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
neural_network_beginner.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))

# 评估结果
print(neural_network_beginner.evaluate(X_test, y_test))

# 第二种构造模型的方式：继承tf.keras.models.Model类，并重写__init__()和call()函数
print('2. use keras model subclassing API to build a neural network for classification')

# 将数据转化为tf.data.Dataset格式
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)


class NeuralNetwork(tf.keras.models.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_1 = tf.keras.layers.Dense(512, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        out = self.output_layer(x)
        return out


neural_network_advanced = NeuralNetwork()

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
        predictions = neural_network_advanced(images)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, neural_network_advanced.trainable_variables)
    optimizer.apply_gradients(zip(gradients, neural_network_advanced.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# 定义测试步
@tf.function
def test_step(images, labels):
    predictions = neural_network_advanced(images)
    loss = loss_function(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


EPOCHS = 5  # 训练轮次

for epoch in range(EPOCHS):
    # 每个epoch开始时将指标重置
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
