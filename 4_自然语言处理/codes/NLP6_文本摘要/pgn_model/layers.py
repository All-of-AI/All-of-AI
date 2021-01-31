import tensorflow as tf
import numpy as np
from utils.config import *


# Encoder类

# input: enc_input(batch_size, enc_seq_len)，编码器的输入
# output: output(batch_size, enc_seq_len, enc_units)，编码器的输出，每个time step都产生一个输出
#         enc_state(batch_size, enc_units)，编码器的最后一步输出
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x):
        # x shape: (batch_size, enc_seq_len)
        embed = self.embedding(x)  # embed shape: (batch_size, enc_seq_len, embedding_dim)
        # 根据输入形状自动get initial state，其内容全部为0。这样了可以不需要单独定义initial state的函数了
        [initial_state] = self.gru.get_initial_state(embed)  # initial_state shape: (batch_size, enc_units)，内容全0

        output, enc_hidden = self.gru(embed, initial_state=[initial_state])
        # output shape: (batch_size, enc_seq_len, enc_units)
        # enc_hidden shape: (batch_size, enc_units)
        return output, enc_hidden


# Bahdanau Attention类

# input: dec_hidden(batch_size, dec_units)，decoder上一步的输出(隐含层向量)，第0步为编码器传来的enc_hidden
#        enc_output(batch_size, enc_seq_len, enc_units)，编码器的输出
#        prev_coverage(batch_size, enc_seq_len, 1)，上一次的coverage(coverage初始值为全0的向量)
# output: context_vector(batch_size, enc_units)，为attention输出的上下文向量
#         squeezed_attention_weights(batch_size, enc_seq_len)，为将最后一维去掉后的attention权重(a)
#         coverage(batch_size, enc_seq_len, 1)，当前time step的coverage向量
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_h = tf.keras.layers.Dense(units)
        self.W_s = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, prev_coverage):
        # query(dec_hidden) shape: (batch_size, dec_units)
        # values(enc_output) shape: (batch_size, enc_seq_len, enc_units)

        # hidden_with_time_axis扩张维度后shape: (batch_size, 1, dec_units)，扩张维度是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 计算带有coverage的attention score
        # self.W_h(hidden_with_time_axis) shape: (batch_size, 1, units)
        # self.W_s(values) shape: (batch_size, enc_seq_len, units)
        # self.W_c(prev_coverage) shape: (batch_size, enc_seq_len, units)

        # score shape: (batch_size, enc_seq_len, 1)
        score = self.V(tf.nn.tanh(self.W_h(hidden_with_time_axis) + self.W_s(values) + self.W_c(prev_coverage)))

        # attention_weights shape: (batch_size, enc_seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # attention weigts的计算与普通seq2seq中的attention一致

        # attention_weights = masked_attention(enc_pad_mask, attention_weights)  # mask层，这里暂不使用

        coverage = attention_weights + prev_coverage  # 当前步的coverage等于当前的attention weights加之前的coverage

        context_vector = attention_weights * values  # context vector的计算与普通seq2seq中的attention一致
        context_vector = tf.reduce_sum(context_vector, axis=1)  # context_vector shape: (batch_size, enc_units)

        # 返回attention_weights前将其最后一维(1)去掉
        return context_vector, tf.squeeze(attention_weights, -1), coverage


# Decoder类，一次调用只decode一步，产生一个单词
# input: x(batch_size, 1)
#        dec_hidden(batch_size, dec_units)
#        enc_output(batch_size, enc_seq_len, enc_units)
#        pre_coverage(batch_size, enc_seq_len, 1)
# output: context_vector(batch_size, enc_units)
#         dec_hidden(batch_size, dec_units)
#         dec_x(batch_size, embedding_dim)
#         pred(batch_size, vocab_size)
#         attention_weights(batch_size, enc_seq_len)
#         coverage(batch_size, enc_seq_len, 1)
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, attention):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention = attention
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, dec_hidden, enc_output, prev_coverage):
        # x shape: (batch_size, 1)
        # dec_hidden shape: (batch_size, dec_units)
        # enc_output shape: (batch_size, enc_seq_len, enc_units)
        # prev_coverage shape: (batch_size, enc_seq_len, 1)

        embed = self.embedding(x)  # embed shape: (batch_size, 1, embedding_dim)，需返回，用于pgen运算

        # 应用GRU单元算出dec_hidden

        dec_output, dec_hidden = self.gru(embed, initial_state=dec_hidden)
        # dec_output (batch_size, 1, dec_units)
        # dec_hidden (batch_size, dec_units)

        # 计算注意力，得到上下文向量，注意力分布，coverage
        # context_vector (batch_size, enc_units)
        # attention_weights (batch_size, enc_seq_len)
        # coverage (batch_size, enc_seq_len, 1)
        context_vector, attention_weights, coverage = self.attention(dec_hidden, enc_output, prev_coverage)

        # 将上一循环的预测结果跟注意力权重值结合在一起来预测vocab的分布
        # dec_output shape: (batch_size, 1, enc_units + dec_units)
        dec_output = tf.concat([dec_output, tf.expand_dims(context_vector, 1)], axis=-1)

        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))  # 将axis=1维度的1去掉
        # dec_output shape: (batch_size, vocab_size)

        # pred shape: (batch_size, enc_units + dec_units)
        pred = self.fc(dec_output)

        return context_vector, dec_hidden, embed, pred, attention_weights, coverage


# 指针网络类，计算Pgen(原文式8)
# input: context_vector(batch_size, enc_units), dec_hidden(batch_size, dec_units), embed(batch_size, 1, embedding_dim)
# output: pgen(batch_size, 1)
class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_h_reduce = tf.keras.layers.Dense(1)  # 用于和一步的context vector相乘
        self.w_s_reduce = tf.keras.layers.Dense(1)  # 用于和decoder一步的hidden相乘
        self.w_x_reduce = tf.keras.layers.Dense(1)  # 用于和decoder一步的input相乘

    def call(self, context_vector, dec_hidden, embed):
        # Pgen的计算公式，context_vector即ht，dec_hidden即st，dec_input即xt
        embed = tf.reshape(embed, (-1, embed.shape[2]))  # 将embed的shape变为(batch_size, embedding_dim)
        pgen = tf.nn.sigmoid(self.w_h_reduce(context_vector) + self.w_s_reduce(dec_hidden) + self.w_x_reduce(embed))
        return pgen  # shape: (batch_size, 1)


# 数据及模型测试
if __name__ == '__main__':
    VOCAB_SIZE = 15000 + 1  # 字典大小
    print('字典大小: ', VOCAB_SIZE)

    # 测试用参数
    EMBEDDING_DIM = 500
    ATTN_UNITS = 20
    ENC_UNITS = 512
    DEC_UNITS = 512
    BATCH_SIZE = 64
    SEQ_LEN = 200

    # 编码器测试
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, ENC_UNITS, BATCH_SIZE)
    example_x = tf.ones(shape=(BATCH_SIZE, SEQ_LEN), dtype=tf.int32)  # 不能用int，要写为tf.int32，二者不兼容
    enc_output, enc_hidden = encoder(example_x)

    print('编码器输出shape: ', enc_output.shape)
    print('编码器最终隐含层向量shape: ', enc_hidden.shape)

    attention = BahdanauAttention(ATTN_UNITS)

    # attention层与解码器测试
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, DEC_UNITS, BATCH_SIZE, attention)

    example_y = tf.ones(shape=(BATCH_SIZE, 1), dtype=tf.int32)  # decoder的一步输入
    dec_hidden = enc_hidden  # decoder的初始隐含层向量
    prev_coverage = tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, 1))  # 初始化coverage为全0的向量

    context_vector, dec_hidden, embed, pred, attention_weights, coverage = \
        decoder(example_y, dec_hidden, enc_output, prev_coverage)

    print('attention层上下文向量shape: ', context_vector.shape)  # (64, 512)
    print('decoder隐含层向量shape: ', dec_hidden.shape)  # (64, 512)
    print('decoder的输入经过embedding层后的shape: ', embed.shape)  # (64, 1, 500)
    print('decoder预测结果shape: ', pred.shape)  # (64, 15001)
    print('attention层的attention权重shape: ', attention_weights.shape)  # (64, 200)
    print('新的coverage的shape: ', coverage.shape)  # (64, 200, 1)

    # 指针网络层测试
    pointer = Pointer()
    pgen = pointer(context_vector, dec_hidden, embed)
    print('指针层输出pgen shape:', pgen.shape)  # (64, 1)
