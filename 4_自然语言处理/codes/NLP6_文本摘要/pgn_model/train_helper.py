import tensorflow as tf
import time
import numpy as np
from pgn_model.batcher import batcher
from pgn_model.loss import calc_loss
from pgn_model.test import test


def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['seq2seq_train_epochs']  # 训练轮次
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])  # 优化器

    # 进行一步训练
    def train_step(target, enc_inp, dec_inp, enc_extended_inp, batch_oov_len, enc_mask, dec_mask, cov_loss_wt):
        with tf.GradientTape() as tape:
            # 正向传播
            final_dist, attentions, coverages = model(enc_inp, dec_inp, enc_extended_inp, batch_oov_len)
            # 计算损失
            batch_loss, log_loss, cov_loss = calc_loss(target, final_dist, dec_mask, attentions, coverages, cov_loss_wt,
                                                       use_coverage=False)  # 先不用use_coverage
        variables = model.trainable_variables  # 直接拿出模型的所有可训练参数
        gradients = tape.gradient(batch_loss, variables)  # 将batch_loss对variables求梯度
        optimizer.apply_gradients(zip(gradients, variables))  # 优化器对梯度进行优化
        return batch_loss, log_loss, cov_loss

    dataset = batcher(vocab, params)  # 获取数据集
    print('获取数据集: ', dataset)
    steps_per_epoch = params['steps_per_epoch']  # 一个epoch需要训练多少步，该参数在制作训练集时写入param

    for epoch in range(epochs):
        params['mode'] = 'train'

        start = time.time()
        total_loss = total_log_loss = total_cov_loss = 0

        for (batch, (enc_data, dec_data)) in enumerate(dataset.take(steps_per_epoch)):
            cov_loss_weight = tf.cast(params["cov_loss_wt"], dtype=tf.float32)

            try:
                batch_oov_len = tf.shape(enc_data["article_oovs"])[1]
            except Exception:
                batch_oov_len = tf.constant(0)

            # 进行一步训练
            batch_loss, log_loss, cov_loss = train_step(dec_data["dec_target"],  # decoder的目标
                                                        enc_data["enc_input"],  # encoder的输入
                                                        dec_data["dec_input"],  # decoder的输入
                                                        enc_data["extended_enc_input"],  # 新加入词表后encoder的输入
                                                        batch_oov_len,  # 一个batch内oov词汇表的大小
                                                        enc_data["enc_mask"],  # encoder的mask
                                                        dec_data["dec_mask"],  # decoder的mask
                                                        cov_loss_weight)  # coverge loss所占权重

            total_loss += batch_loss
            total_log_loss += log_loss
            total_cov_loss += cov_loss

            if batch % 20 == 0:  # 每隔10个batch打印一次训练信息
                print('Epoch {} Batch {} batch_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'
                      .format(epoch + 1, batch, batch_loss.numpy(), log_loss, cov_loss))

        if (epoch + 1) % 1 == 0:  # 每个epoch保存一次模型
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            # 学习率衰减
            lr = params["learning_rate"] * np.power(0.95, epoch + 1)
            # 更新优化器的学习率
            optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            assert lr == optimizer.get_config()["learning_rate"]
            print("衰减后的学习率为: ", optimizer.get_config()["learning_rate"])

        # 打印一个epoch的信息，包括总loss、log loss以及coverage loss三种损失
        print('Epoch {} loss {:.4f} log loss {:.4f} cov loss {:.4f}'.format(epoch + 1,
                                                                            total_loss / steps_per_epoch,
                                                                            total_log_loss / steps_per_epoch,
                                                                            total_cov_loss / steps_per_epoch))
        # 打印一个epoch耗费的时间
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        # 每训练完一个epoch后，进行一次测试
        print('进行一次测试')
        test(params)
