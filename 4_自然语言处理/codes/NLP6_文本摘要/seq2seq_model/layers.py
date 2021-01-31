import tensorflow as tf
from utils.gpu_utils import config_gpu


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,  # 在输出序列中，返回单个hidden state值还是返回全部time step的hidden state值
                                       return_state=True,  # return_state表明是否返回最后一个状态
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # x shape: (batch_size, enc_seq_len, embed_size)
        output, state = self.gru(x, initial_state=hidden)  # output shape: (batch_size, enc_seq_len, enc_units)
        # output shape: (batch_size, enc_seq_len, enc_units)
        # state shape: (batch_size, enc_units)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))  # 初始隐含层向量shape: (batch_size, enc_units)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query为上次的decoder隐藏层，shape: (batch_size, dec_units)
        # values为编码器的编码结果enc_output，shape: (batch_size, enc_seq_len, enc_units)

        # hidden_with_time_axis扩张维度后shape: (batch_size, 1, dec_units)，扩张维度是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 在应用self.V之前，张量的形状是(batch_size, enc_seq_len, attention_units)
        # 得到score的shape: (batch_size, enc_seq_len, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))  # 使用了运算的广播性

        # 注意力权重，是score经过softmax，但是要作用在第一个轴上(seq_len的轴)，不改变shape(batch_size, enc_seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # (batch_size, enc_seq_len, 1) * (batch_size, enc_seq_len, enc_units)，广播，encoder unit的每个位置都对应相乘
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)  # 在编码器长度这一维度上求和
        # context_vector求和之后的shape: (batch_size, enc_units)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)  # 加了softmax，loss不收敛

    def call(self, x, state, context_vector):
        # 使用上次的隐藏层(第一次使用编码器隐藏层)、编码器输出计算注意力权重
        # enc_output shape: (batch_size, enc_seq_len, hidden_size)

        # x shape after passing through embedding: (batch_size, 1, embedding_dim)，1指的是一次只解码一个单词
        x = self.embedding(x)

        output, state = self.gru(x, initial_state=state)  # 将上一步运行解码器得到的state作为下一步解码器的初始state
        # output shape: (batch_size, 1, hidden_size)
        # state shape: (batch_size, hidden_size)

        # 将本步解码器的GRU输出和上下文向量结合，用作输出预测
        output = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
        # output连接context vector后的shape: (batch_size, 1, embedding_dim + hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))  # 将axis=1维度的1去掉
        # output shape: (batch_size, vocab_size)

        prediction = self.fc(output)

        return prediction, state


# 测试网络的层是否正常运行
if __name__ == '__main__':
    # GPU资源配置
    # config_gpu()

    # 测试用参数
    VOCAB_SIZE = 15000 + 1
    EXAMPLE_INPUT_SEQUENCE_LEN = 200
    BATCH_SIZE = 64
    EMBEDDING_DIM = 500
    GRU_UNITS = 512
    ATTENTION_UNITS = 20

    # 编码器结构
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, GRU_UNITS, BATCH_SIZE)
    print('运行编码器')

    # 测试用的一条输入序列
    example_input_batch = tf.ones(shape=(BATCH_SIZE, EXAMPLE_INPUT_SEQUENCE_LEN), dtype=tf.int32)
    # 编码器的初始隐藏层向量
    sample_hidden = encoder.initialize_hidden_state()

    # 编码器的输出以及最终隐藏层的状态
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    # Attention层
    print('运行attention层')
    attention_layer = BahdanauAttention(ATTENTION_UNITS)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    # 解码器结构
    print('运行解码器')
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, GRU_UNITS, BATCH_SIZE)
    pred, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, context_vector)
