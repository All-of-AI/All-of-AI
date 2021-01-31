import tensorflow as tf

from layers import Encoder, BahdanauAttention, Decoder
from utils.gpu_utils import config_gpu


class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.params = params
        self.encoder = Encoder(params["vocab_size"], params["embed_size"], params["enc_units"], params["batch_size"])
        self.attention = BahdanauAttention(params["attn_units"])
        self.decoder = Decoder(params["vocab_size"], params["embed_size"], params["dec_units"], params["batch_size"])

    # call实质上是在调用解码器，因为需要囊括注意力机制，直接封装到call中。要调用编码器直接encoder()即可
    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        # 这里的dec_input实质是(batch_size, 1)大小的<START>
        predictions = []

        # 拿编码器的输出和最终隐含层向量来计算
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)

        for t in range(1, dec_target.shape[1]):
            # dec_input (batch_size, 1)；dec_hidden (batch_size, dec_units)
            pred, dec_hidden = self.decoder(dec_input, dec_hidden, context_vector)

            context_vector, attention_weights = self.attention(dec_hidden, enc_output)

            # 使用teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)

        # 必须stack将一列一列(每一次predict)堆在一起再返回
        return tf.stack(predictions, 1), dec_hidden  # decoder output (batch_size, dec_seq_len, vocab_size)


# 测试模型是否运行成功
if __name__ == '__main__':
    # GPU资源配置
    # config_gpu()

    # 测试用参数
    vocab_size = 15000 + 1
    batch_size = 64
    input_seq_len = 200

    params = {"vocab_size": vocab_size, "embed_size": 500, "enc_units": 512, "attn_units": 20, "dec_units": 512,
              "batch_size": batch_size}

    model = Seq2Seq(params)

    # sample_input
    sample_input_batch = tf.ones(shape=(batch_size, input_seq_len), dtype=tf.int32)

    # sample inital hidden vector
    sample_hidden = model.encoder.initialize_hidden_state()

    sample_output, sample_hidden = model.encoder(sample_input_batch, sample_hidden)

    print('Encoder output shape: (batch_size, enc_seq_len, enc_units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch_size, enc_units) {}'.format(sample_hidden.shape))

    context_vector, attention_weights = model.attention(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch_size, enc_units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, = model.decoder(tf.random.uniform((batch_size, 1)), sample_hidden, context_vector)

    print('Decoder output shape: (batch_size, vocab_size) {}'.format(sample_decoder_output.shape))
    # 这里仅测试一步，没有用到dec_seq_len
