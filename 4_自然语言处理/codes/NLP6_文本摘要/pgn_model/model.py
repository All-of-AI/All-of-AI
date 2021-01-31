import tensorflow as tf
from pgn_model.layers import Encoder, BahdanauAttention, Decoder, Pointer


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.params = params
        self.encoder = Encoder(params["vocab_size"], params["embed_size"], params["enc_units"], params["batch_size"])
        self.attention = BahdanauAttention(units=params["attn_units"])
        self.decoder = Decoder(params["vocab_size"], params["embed_size"], params["dec_units"], params["batch_size"],
                               self.attention)
        self.pointer = Pointer()

    def call(self, enc_inp, dec_inp, enc_extended_inp, batch_oov_len):
        # enc_inp shape: (batch_size, enc_seq_len)
        # dec_inp shape: (batch_size, dec_seq_len)
        # enc_extended_inp shape: (batch_size, enc_seq_len)
        # batch_oov_len: scalar

        predictions = []
        attentions = []
        p_gens = []
        coverages = []

        # 调用编码器
        enc_output, enc_hidden = self.encoder(enc_inp)

        # 解码器的初始隐含层向量为编码器最后一步输出的隐含层向量
        dec_hidden = enc_hidden

        # 初始化第一步coverage为全为0的tensor，形状为(batch_size, enc_seq_len, 1)
        prev_coverage = tf.zeros((enc_output.shape[0], enc_output.shape[1], 1))

        # 调用dec_seq_len步解码器
        for t in tf.range(dec_inp.shape[1]):
            context_vector, dec_hidden, embed, pred, attention_weights, prev_coverage = \
                self.decoder(dec_inp[:, t:t+1], dec_hidden, enc_output, prev_coverage)  # teacher forcing

            p_gen = self.pointer(context_vector, dec_hidden, embed)

            # 每次解码后把相应数据写入结果数组
            predictions.append(pred)
            attentions.append(attention_weights)
            p_gens.append(p_gen)
            coverages.append(prev_coverage)

        # 因为解码是一步一步来的，所以将结果按列堆积
        predictions = tf.stack(predictions, axis=1)  # stack后的shape: (batch_size, dec_seq_len, vocab_size)
        attentions = tf.stack(attentions, axis=1)  # stack后的shape: (batch_size,dec_seq_len, enc_seq_len)
        p_gens = tf.stack(p_gens, axis=1)  # stack后的shape: (batch_size, dec_seq_len, 1)
        coverages = tf.stack(coverages, axis=1)

        coverages = tf.squeeze(coverages, -1)  # coverages去掉最后一个为1的轴

        # 计算最终的分布形式
        final_distribution = calc_final_distribution(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len,
                                                     self.params["vocab_size"], self.params["batch_size"])

        # final_dist shape: (batch_size, dec_len, vocab_size + batch_oov_len)
        # attentions和coverages的shape: (batch_size, dec_len, enc_len)
        return final_distribution, attentions, coverages


def calc_final_distribution(enc_batch_extend, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    按照公式，计算单词的最终分布
    """
    # 先计算公式的左半部分，vocab_dists就是解码器预测的predictions
    vocab_dists_pgn = vocab_dists * p_gens  # shape: (batch_size, dec_seq_len, vocab_size)
    # 根据oov表的长度补齐原词表
    # extra_zeros (batch_size, dec_seq_len, batch_oov_len)
    if batch_oov_len != 0:
        extra_zeros = tf.zeros((batch_size, p_gens.shape[1], batch_oov_len))
        # 拼接后公式的左半部分完成了，给公式右半部分留出了额外batch_oov_len大小的0
        # vocab_dists_extended (batch_size, dec_seq_len, vocab_size + batch_oov_len)
        vocab_dists_extended = tf.concat([vocab_dists_pgn, extra_zeros], axis=-1)

    # 公式右半部分
    # 乘以权重后的注意力
    # attn_dists_pgn (batch_size, dec_seq_len, enc_seq_len)
    attn_dists_pgn = attn_dists * (1 - p_gens)

    # 拓展后的长度
    extended_vocab_size = vocab_size + batch_oov_len

    # 要更新的数组attn_dists_pgn
    # 更新之后数组的形状与 公式左半部分一致
    # shape=(batch_size, dec_seq_len, vocab_size + batch_oov_len)
    shape = vocab_dists_extended.shape

    enc_len = tf.shape(enc_batch_extend)[1]
    dec_len = tf.shape(vocab_dists_extended)[1]

    # batch_nums (batch_size, )
    batch_nums = tf.range(0, limit=batch_size)
    # batch_nums (batch_size, 1)
    batch_nums = tf.expand_dims(batch_nums, 1)
    # batch_nums (batch_size, 1, 1)
    batch_nums = tf.expand_dims(batch_nums, 2)

    # tile在第1和第2个维度上分别复制batch_nums dec_seq_len,enc_seq_len
    # batch_nums (batch_size, dec_seq_len, enc_seq_len)
    batch_nums = tf.tile(batch_nums, [1, dec_len, enc_len])

    # (dec_len, )
    dec_len_nums = tf.range(0, limit=dec_len)
    # (1, dec_len)
    dec_len_nums = tf.expand_dims(dec_len_nums, 0)
    # (1, dec_len, 1)
    dec_len_nums = tf.expand_dims(dec_len_nums, 2)
    # tile是用来在不同维度上复制张量的
    # dec_len_nums (batch_size, dec_len, enc_len)
    dec_len_nums = tf.tile(dec_len_nums, [batch_size, 1, enc_len])

    # enc_batch_extend_vocab_expand (batch_size, 1, enc_len)
    enc_batch_extend_vocab_expand = tf.expand_dims(enc_batch_extend, 1)
    # enc_batch_extend_vocab_expand (batch_size, dec_len, enc_len)
    enc_batch_extend_vocab_expand = tf.tile(enc_batch_extend_vocab_expand, [1, dec_len, 1])

    # 因为要scatter到一个3D tensor上，所以最后一维是3
    # indices (batch_size, dec_len, enc_len, 3)
    indices = tf.stack((batch_nums, dec_len_nums, enc_batch_extend_vocab_expand), axis=3)

    # 开始更新
    attn_dists_projected = tf.scatter_nd(indices, attn_dists_pgn, shape)
    # 至此完成了公式的右半部分

    # 计算最终分布
    final_distribution = vocab_dists_extended + attn_dists_projected

    return final_distribution
