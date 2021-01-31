import tensorflow as tf
from tqdm import tqdm
from pgn_model.model import calc_final_distribution


def greedy_decode(model, dataset, vocab, params):
    batch_size = params["batch_size"]
    results = []  # 存储结果

    sample_size = 20000
    steps_epoch = sample_size // batch_size  # 若20000不能整除batch_size，结果需要+1

    ds = iter(dataset)
    for _ in tqdm(range(steps_epoch)):  # [0, steps_epoch)
        enc_data, dec_data = next(ds)

        results += batch_greedy_decode(model, enc_data, vocab, params)
    return results


def decode_one_step(params, model, enc_extended_inp, batch_oov_len, dec_input, dec_hidden, enc_output,
                    prev_coverage, batch_size):
    # 开始decoder
    context_vector, dec_hidden, dec_x, pred, attn, coverage = model.decoder(dec_input, dec_hidden, enc_output,
                                                                            prev_coverage)

    # 计算p_gen
    p_gen = model.pointer(context_vector, dec_hidden, dec_x)

    # 保证pred attn p_gen的参数为三维
    final_distributinon = calc_final_distribution(enc_extended_inp,
                                                  tf.expand_dims(pred, 1),
                                                  tf.expand_dims(attn, 1),
                                                  tf.expand_dims(p_gen, 1),
                                                  batch_oov_len,
                                                  params["vocab_size"],
                                                  batch_size)

    return final_distributinon, dec_hidden, coverage


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data  # shape: (batch_size, enc_seq_len)

    enc_output, enc_hidden = model.encoder(inputs)  # 经过编码器
    dec_hidden = enc_hidden  # 解码器的初始向量为编码器最后一步的隐含层向量
    # decoder输入初始shape: (batch_size, 1)
    dec_input = tf.expand_dims([vocab.word2id['start']] * batch_size, 1)

    # 获取oov_len
    try:
        batch_oov_len = tf.shape(enc_data["article_oovs"])[1]
    except Exception:
        batch_oov_len = tf.constant(0)

    coverage = tf.zeros((enc_output.shape[0], enc_output.shape[1], 1))  # coverage初始值

    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        final_dist, dec_hidden, coverage = decode_one_step(params, model, enc_data["extended_enc_input"], batch_oov_len,
                                                           dec_input, dec_hidden, enc_output, coverage, batch_size)

        # index转换
        final_dist = tf.squeeze(final_dist, axis=1)  # (batch_size, vocab_size+batch_oov_len)
        predicted_ids = tf.argmax(final_dist, axis=1)  # (batch_size, )

        for index, predicted_id in enumerate(predicted_ids.numpy()):
            if predicted_id < vocab.size():
                predicts[index] += vocab.id_to_word(predicted_id) + ' '
            else:
                x = enc_data['article_oovs'][index, predicted_id - vocab.size()]
                x = str(x.numpy(), 'utf-8')
                predicts[index] += x + ' '

        # 将未知词汇(id大于等于词表大小)转换为<UNK>对应的id
        predicted_ids_list = []
        for i in range(len(predicted_ids)):
            if predicted_ids[i] >= vocab.size():
                predicted_ids_list.append(vocab.word_to_id('<UNK>'))
            else:
                predicted_ids_list.append(predicted_ids[i])

        predicted_ids = tf.convert_to_tensor(predicted_ids_list)

        dec_input = tf.expand_dims(predicted_ids, 1)  # 自回归

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了，截断
        if 'end' in predict:
            # 截断end
            predict = predict[:predict.index('end')]
        # 保存结果
        results.append(predict)
    return results
