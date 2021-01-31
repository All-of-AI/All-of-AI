import tensorflow as tf
import time
from seq2seq_model.batcher import train_batch_generator
from seq2seq_model.test import test


def train_model(model, vocab, params, checkpoint_manager):
    # 载入参数：seq2seq模型的训练轮次以及batch大小
    epochs = params['seq2seq_train_epochs']
    batch_size = params['batch_size']

    pad_index = 0
    nuk_index = vocab['<UNK>']
    start_index = vocab['start']

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params['learning_rate'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        pad_mask = tf.math.equal(real, pad_index)  # mask掉训练标签中的<pad>
        nuk_mask = tf.math.equal(real, nuk_index)  # mask掉训练标签中的<unk>
        mask = tf.math.logical_not(tf.math.logical_or(pad_mask, nuk_mask))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # 训练
    @tf.function
    def train_step(enc_input, dec_target):
        with tf.GradientTape() as tape:
            initial_hidden_state = model.encoder.initialize_hidden_state()  # 初始化隐藏层向量
            enc_output, enc_hidden = model.encoder(enc_input, initial_hidden_state)  # 调用编码器
            # 解码器的第一步输入为开始标签<START>
            dec_input = tf.expand_dims([start_index] * batch_size, 1)
            # 解码器第一步的隐含状态为编码器最终输出的状态
            dec_hidden = enc_hidden
            # 逐个预测序列
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)

            batch_loss = loss_function(dec_target[:, 1:], predictions)  # 形状均为(batch, len(dec_target)-1)

        # 训练参数
        variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables
        # 计算梯度
        gradients = tape.gradient(batch_loss, variables)
        # 优化器优化
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    # 读取数据
    print('读取数据')
    dataset, steps_per_epoch = train_batch_generator(batch_size)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 20 == 0:  # 每20个batch打印一次训练信息
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

        if (epoch + 1) % 1 == 0:  # 每两个epoch保存一次模型
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        # 打印一个epoch的训练信息
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        # 打印一个epoch所用时间
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # 每训练完一个epoch后，进行一次测试
        print('进行一次测试')
        test(params)
