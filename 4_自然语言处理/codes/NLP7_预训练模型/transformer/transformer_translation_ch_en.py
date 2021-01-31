import tensorflow as tf
import unicodedata
import re
import numpy as np
import io
import time
import jieba

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformer.langconv import Converter


def config_gpu():
    """
    If memory growth is enabled for a PhysicalDevice, the runtime initialization will not allocate all
    memory on the device. Memory growth cannot be configured on a PhysicalDevice with virtual devices configured.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


# 配置GPU
# config_gpu()

# 原始数据集路径
path_to_file = 'data_ch_en.txt'


# 数据预处理
# 将 unicode 文件转换为 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_english(w):
    w = unicode_to_ascii(w.lower().strip())
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" ]+', " ", w)
    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    w = '<start> ' + w + ' <end>'
    return w


# 预处理测试
en_sentence = "May I borrow this book?"
print('句子的预处理: ')
print(en_sentence, '  ->  ', preprocess_english(en_sentence))


def convert(text):
    # 将繁体中文转换为简体中文
    rule = 'zh-hans'
    return Converter(rule).convert(text)


print(convert('你的寶寶，什麼時候開始說話的？'))


# 构造数据集
# 1. 提取中英文
# 2. 清理英语句子，中文分词，添加开始和结束符
# 3. 返回这样格式的单词对：[英语, 汉语]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')[:2]] for l in lines[:num_examples]]

    for i in range(len(word_pairs)):
        word_pairs[i][0] = preprocess_english(word_pairs[i][0])  # 将英文进行简单的预处理
        word_pairs[i][1] = "<start> " + " ".join(
            jieba.cut(convert(word_pairs[i][1]), cut_all=False)) + " <end>"  # 将中文从繁体转换为简体，并加入开始结束符号

    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  # 后部补0
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)  # target为英文，input为中文

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    # 返回两种语言的tensor，以及对应的tokenizer
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# 限制数据集的大小以加快实验速度
num_examples = 30000
# 读取input(中文)向量、output(英语)向量以及两个语言的tokenizer
input_tensor, target_tensor, tokenizer_ch, tokenizer_en = load_dataset(path_to_file, num_examples)
print(tokenizer_ch.word_index)
print(tokenizer_en.word_index)
# 计算目标张量的最大长度 （max_length）
max_length_inp, max_length_targ = max_length(input_tensor), max_length(target_tensor)
print('所有语料中，汉语的最大长度为: ', max_length_inp, '英语的最大长度为: ', max_length_targ)

# 采用80-20的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)

# 显示长度
print('训练和测试集的长度(均为ndarray)，每一条代表一条句子index序列): ', len(input_tensor_train), len(target_tensor_train),
      len(input_tensor_val), len(target_tensor_val))

# 创建tf.data.Data数据集
train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
train_dataset = train_dataset.batch(64, drop_remainder=True)
print('数据集: ', train_dataset)

example_input_batch, example_target_batch = next(iter(train_dataset))
print('一个批次的训练数据和标签的shape: ')
print(example_input_batch.shape, '\n', example_target_batch.shape)
# 至此，数据集处理完毕
print('至此，数据集处理完毕\n')


# 位置编码(Positional encoding)
# 因为该模型并不包括任何的循环或卷积，所以模型添加了位置编码，为模型提供一些关于单词在句子中相对位置的信息

# 位置编码向量产生后被加到embedding向量中。嵌入表示一个d维空间的标记，在d维空间中有着相似含义的标记会离彼此更近。但是，嵌入并没有对
# 在一句话中的词的相对位置进行编码。因此，当加上位置编码后，词将基于它们含义的相似度以及它们在句子中的位置，在d维空间中离彼此更近
def get_angles(pos, i, d_model):
    # pos为单词在句子中的绝对位置，i指的是单词embedding向量的第i维
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    # d_model是位置编码的维度，也是embedding向量的维度
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # 将sin应用于数组中的偶数索引2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 将cos应用于数组中的奇数索引2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


pos_encoding = positional_encoding(50, 512)
print('位置编码的shape: ', pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()


# 遮挡(Masking)
# 遮挡一批序列中所有的填充标记(pad tokens)。这确保了模型不会将填充作为输入。该mask表明填充值0出现的位置：
# 在这些位置 mask 输出 1，否则输出 0。
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到注意力对数(logits)。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
print('[[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]对0进行mask后: ', create_padding_mask(x))


# 前瞻遮挡(look-ahead mask)用于遮挡一个序列中的后续标记(future tokens)。换句话说，该 mask 表明了不应该使用的条目。
# 这意味着要预测第三个词，将仅使用第一个和第二个词。与此类似，预测第四个词，仅使用第一个，第二个和第三个词，依此类推。
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
print('tf.random.uniform((1, 3))的前瞻遮盖: ', temp)


def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v必须具有匹配的后置维度。
    k, v必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然mask根据其类型（填充或前瞻）有不同的形状，但是mask必须能进行广播转换以便求和。
    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth)
      mask: Float 张量，其形状能转换成(..., seq_len_q, seq_len_k)，默认为None。
    返回值:
      输出，注意力权重
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将mask加入到缩放后的张量上
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax在最后一个轴（seq_len_k）上归一化，因此分数相加等于1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)
    return output, attention_weights


# 当softmax在K上进行归一化后，它的值决定了分配到Q的重要程度
# 输出表示注意力权重和 V(数值)向量的乘积。这确保了要关注的词保持原样，而无关的词将被清除掉
def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


np.set_printoptions(suppress=True)

temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)  # (4, 2)

# 这条query符合第二个key，因此返回了第二个value
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)


# 多头注意力(Multi-head attention)
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        输入维度: (batch_size, seq_len, d_model)
        分拆最后一个维度到(num_heads, depth)，转置结果使得输出维度为(batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):  # seq_len_q == seq_len_k == seq_len_v
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # scaled_attention shape after transpose: (batch_size, seq_len_q, num_heads, depth)

        # 将多个attention head进行concat，由于是在一个数组中的，只需reshape即可
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # concat_attention shape: (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
print(out.shape, attn.shape)


# 点式前馈网络(Point wise feed forward network)
# 点式前馈网络由两层全联接层组成，两层之间有一个ReLU激活函数
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # input shape: (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # output shape: (batch_size, seq_len, d_model)
    ])


sample_ffn = point_wise_feed_forward_network(512, 2048)
print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)


# 编码器层
# 每个编码器层包括以下子层：
# 1. 多头注意力(有填充遮挡)
# 2. 点式前馈网络(Point wise feed forward networks)
# 每个子层在其周围有一个残差连接，然后进行层归一化。残差连接有助于避免深度网络中的梯度消失问题。

# 每个子层的输出是 LayerNorm(x + Sublayer(x))。归一化是在 d_model(最后一个)维度完成的。
# Transformer中有N个编码器层。

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


sample_encoder_layer = EncoderLayer(512, 8, 2048)  # 定义编码器层的一个实例
sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
print('编码器层的输出shape: ', sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)


# 解码器层(Decoder layer)
# 每个解码器层包括以下子层：

# 1. 遮挡的多头注意力(前瞻遮挡和填充遮挡)
# 2. 多头注意力(用填充遮挡)。V(数值)和K(主键)接收编码器输出作为输入。Q(请求)接收遮挡的多头注意力子层的输出。
# 3. 点式前馈网络
# 每个子层在其周围有一个残差连接，然后进行层归一化。每个子层的输出是LayerNorm(x + Sublayer(x))。归一化是在d_model(最后一个)维度完成的

# Transformer中共有N个解码器层。

# 当Q接收到解码器的第一个注意力块的输出，并且K接收到编码器的输出时，注意力权重表示根据编码器的输出赋予解码器输入的重要性
# 换一种说法，解码器通过查看编码器输出和对其自身输出的自注意力，预测下一个词。参看按比缩放的点积注意力部分的演示
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                               padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# 测试解码器层
sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
                                                         False, None, None)

print('解码器层的输出shape: ', sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)


# 编码器(Encoder)包括：
# 1. 输入嵌入(Input Embedding)
# 2. 位置编码(Positional Encoding)
# 3. N个编码器层(encoder layers)
# 输入经过嵌入(embedding)后，该嵌入与位置编码相加。该加法结果的输出是编码器层的输入。编码器的输出是解码器的输入
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)

sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)), training=False, mask=None)

print('编码器的输出shape: ', sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)


# 解码器(Decoder)包括：
# 输出嵌入(Output Embedding)
# 位置编码(Positional Encoding)
# N 个解码器层(decoder layers)
# 目标(target)经过一个嵌入后，该嵌入和位置编码相加。该加法结果是解码器层的输入。解码器的输出是最后的线性层的输入
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)

output, attn = sample_decoder(tf.random.uniform((64, 26)),
                              enc_output=sample_encoder_output,
                              training=False, look_ahead_mask=None,
                              padding_mask=None)

print('解码器的输出shape: ', output.shape, attn['decoder_layer2_block2'].shape)


# 创建Transformer
# Transformer包括编码器，解码器和最后的线性层。解码器的输出是线性层的输入，返回线性层的输出。
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)  # 编码器
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)  # 解码器
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # 用于预测的最后一层

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                                 target_vocab_size=8000, pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 62))
temp_target = tf.random.uniform((64, 26))

fn_out, _ = sample_transformer(temp_input, temp_target, training=False, enc_padding_mask=None, look_ahead_mask=None,
                               dec_padding_mask=None)

print('Transformer输出的shape: ', fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

# 配置超参数（hyperparameters）
# 为了让本示例小且相对较快，已经减小了num_layers、 d_model和dff的值
# Transformer的基础模型使用的数值为：num_layers=6，d_model = 512，dff = 2048。关于所有其他版本的Transformer，请查阅论文
# Note：通过改变以下数值，您可以获得在许多任务上达到最先进水平的模型
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = len(tokenizer_ch.word_index) + 2
target_vocab_size = len(tokenizer_en.word_index) + 2
dropout_rate = 0.1


# 优化器(Optimizer)
# 根据论文中的公式，将Adam优化器与自定义的学习速率调度程序(scheduler)配合使用
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # 定义优化器

temp_learning_rate_schedule = CustomSchedule(d_model)

plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))

# 损失函数与指标
# 由于目标序列是填充过的，因此在计算损失函数时，应用填充遮挡非常重要
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# 训练与检查点(Training and checkpointing)
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size, pe_target=target_vocab_size, rate=dropout_rate)


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用
    # 该填充遮挡用于遮挡编码器的输出
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用
    # 用于填充(pad)和遮挡(mask)解码器获取到的输入的后续标记(future tokens)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# 创建检查点的路径和检查点管理器(manager)。这将用于在每n个周期(epochs)保存检查点
checkpoint_path = "checkpoints/ch_en"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('模型从检查点中恢复')

# 目标(target)被分成了 tar_inp 和 tar_real。tar_inp作为输入传递到解码器。tar_real是位移了1的同一个输入：在tar_inp中的每个位置，tar_real包含了应该被预测到的下一个标记

# 例如，sentence = "SOS A lion in the jungle is sleeping EOS"
# tar_inp = "SOS A lion in the jungle is sleeping"
# tar_real = "A lion in the jungle is sleeping EOS"

# Transformer是一个自回归模型：它一次作一个部分的预测，然后使用到目前为止的自身的输出来决定下一步要做什么。
# 在训练过程中，本示例使用了teacher-forcing的方法。无论模型在当前时间步骤下预测出什么，teacher-forcing方法都会将真实的输出传递到下一个时间步骤上。
# 当 transformer预测每个词时，自注意力功能使它能够查看输入序列中前面的单词，从而更好地预测下一个单词。
# 为了防止模型在期望的输出上达到峰值，模型使用了前瞻遮挡(look-ahead mask)。

# 该@tf.function将追踪-编译train_step到TF图中，以便更快地执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小(最后一批次较小)导致的再追踪，使用input_signature指定更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]  # teacher forcing

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train(EPOCHS):
    print('模型训练')

    # 中文作为输入语言，英语为目标语言
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = tokenizer_ch.word_index['<start>']
    end_token = tokenizer_ch.word_index['<end>']

    inp_sentence_list = []
    for i in inp_sentence.split(' '):
        if i in tokenizer_ch.word_index.keys():
            inp_sentence_list.append(tokenizer_ch.word_index[i])

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = [start_token] + inp_sentence_list + [end_token]

    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the english <start> token.
    decoder_input = [tokenizer_en.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)  # 变为二维数据

    for i in range(50):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # concatentate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(sentence):
    print('进行一次翻译')
    result, attention_weights = evaluate(sentence)

    result = list(result.numpy())
    result = result[1:]  # 去掉<start>开始标记
    for i in range(len(result)):
        if result[i] == tokenizer_en.word_index['<end>']:
            result = result[:i]  # 去掉<end>结束标记
            break

    predicted_sentence = [tokenizer_en.index_word[i] for i in result if i <= len(tokenizer_en.word_index)]

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(" ".join(predicted_sentence)))


if __name__ == '__main__':
    mode = 'translate'  # train/translate

    if mode == 'train':
        train(EPOCHS=20)

    elif mode == 'translate':
        ch_sent = "他的爸爸是一名医生。"
        en_sent = "His father is a doctor ."

        ch_sent_cut = " ".join(jieba.cut(ch_sent))

        ch_sent_words = ch_sent_cut.split(' ')
        ch_sent_in_vocab = []
        for word in ch_sent_words:
            if word in tokenizer_ch.word_index.keys():
                ch_sent_in_vocab.append(word)

        ch_sent_in_vocab = " ".join(ch_sent_in_vocab)

        translate(ch_sent_cut)
        print("Input in vocab: ", ch_sent_in_vocab)
        print("Real translation: ", en_sent)
