from utils.load_data import load_train_dataset, load_test_dataset
import tensorflow as tf


def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, sample_num=None):
    """
    加载训练集为tf.seq2seq_data.Dataset格式
    :param batch_size: batch大小
    :param max_enc_len: 样本最大长度
    :param max_dec_len: 标签最大长度
    :param sample_num: 限定样本个数大小
    :return: 训练数据集, 一个epoch取多少个batch
    """
    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)
    if sample_num:
        train_X = train_X[:sample_num]
        train_Y = train_Y[:sample_num]
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    dataset = dataset.batch(batch_size, drop_remainder=True)  # 丢弃最后不足batch_size的训练数据
    steps_per_epoch = len(train_X) // batch_size
    return dataset, steps_per_epoch
