import tensorflow as tf
from tqdm import tqdm


def greedy_decode(model, test_X, vocab, reverse_vocab, params):
    # 存储结果
    batch_size = params['batch_size']
    results = []

    sample_size = len(test_X)
    # batch操作轮数math.ceil向上取整小数+1，因为最后一个batch可能不足一个batch size大小 ,但是依然需要计算
    steps_epoch = sample_size // batch_size  # + 1

    # [0, steps_epoch)
    for i in tqdm(range(steps_epoch)):  # tqdm用于展示进度条
        batch_data = test_X[i * batch_size:(i + 1) * batch_size]
        results += batch_greedy_decode(model, batch_data, vocab, reverse_vocab, params)

    return results


def batch_greedy_decode(model, batch_data, vocab, reverse_vocab, params):
    # 判断输入长度
    batch_size = len(batch_data)
    # 开辟结果存储list
    predicts = [''] * batch_size

    # 转变为tensor
    inputs = tf.convert_to_tensor(batch_data)

    # 注意这里的batch_size与config中的batch_size可能不一致，原因是最后一个batch可能不是64个数据，因此应当按以下形式初始化隐藏层
    initial_hidden_state = tf.zeros((batch_size, model.encoder.enc_units))

    # 编码器运算
    enc_output, enc_hidden = model.encoder(inputs, initial_hidden_state)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([vocab['start']] * batch_size, 1)

    context_vector, _ = model.attention(dec_hidden, enc_output)

    # 测试阶段，采用自回归的方式生成结果
    for t in range(params['max_dec_len']):
        # 计算上下文
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        # 单步预测
        predictions, dec_hidden = model.decoder(dec_input, dec_hidden, context_vector)
        # id转换，贪婪搜索
        predicted_ids = tf.argmax(predictions, axis=1).numpy()

        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += reverse_vocab[predicted_id] + ' '

        dec_input = tf.expand_dims(predicted_ids, 1)  # 自回归

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if 'end' in predict:
            # 截断结束符后的内容
            predict = predict[:predict.index('end')]
        results.append(predict)
    return results
