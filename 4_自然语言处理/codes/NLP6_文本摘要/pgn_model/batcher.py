import tensorflow as tf
from utils.config import *
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params


class Vocab:
    def __init__(self, vocab_file):
        """
        Vocab对象，vocab基本操作封装
        :param vocab_file: Vocab存储路径
        """
        params = get_params()
        self.count = params['vocab_size']
        self.word2id, self.id2word = self.load_vocab(vocab_file)  # 正向字典和反向字典

    def load_vocab(self, file_path):
        """
        读取字典
        :param file_path: 字典文件路径
        :return: 返回读取后的字典和反向字典
        """
        vocab = {}
        inverse_vocab = {}
        for line in open(file_path, "r", encoding='utf-8').readlines():
            word, index = line.strip().split("\t")
            index = int(index)  # 将字符串类型转为整型

            # 控制字典大小
            if index < self.size():
                vocab[word] = index
                inverse_vocab[index] = word
            else:
                return vocab, inverse_vocab

        return vocab, inverse_vocab

    def word_to_id(self, word):
        if word == '<START>':
            return self.word2id['start']
        elif word == '<END>':
            return self.word2id['end']
        elif word not in self.word2id:
            return self.word2id['<UNK>']
        return self.word2id[word]

    def id_to_word(self, word_id):  # 仅在测试时使用
        if word_id not in self.id2word:
            raise ValueError('id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        # params['vocab_size']，有效单词个数 + 1
        return self.count


def article_to_ids(article_words, vocab):
    """
    将artivle(X)的单词列表转换为index列表以及oovs列表
    :param article_words: 数据样本的单词列表，格式为['新能源', '车', '最大', '短板']
    :param vocab: 字典
    """
    ids = []  # 存放最终的index结果
    oovs = []  # 存放最终的OOV词汇结果
    unk_id = vocab.word_to_id('<UNK>')  # 未知词汇对应的index

    for w in article_words:
        # 开始和结束标记符单独处理
        if w == '<START>':
            ids.append(vocab.word2id['start'])
        elif w == '<END>':
            ids.append(vocab.word2id['end'])
        # 其他单词(包括词典内的词及OOV词汇)
        else:
            i = vocab.word_to_id(w)  # 通过词典获取当前单词的index
            if i == unk_id:  # 如果发现oov词
                if w not in oovs:  # 如果oov列表还没有该oov词
                    oovs.append(w)  # 该oov词加入oov列表
                oov_num = oovs.index(w)  # 该句第一个oov词 oov_num=0, 第二个oov词 oov_num=1
                ids.append(vocab.size() + oov_num)  # 加入该词
            else:
                ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    """
    将abstract(Y)的单词列表转换为index列表
    :param abstract_words: 数据标签的单词列表，格式为['新能源', '车', '最大', '短板']
    :param vocab: 字典
    :param article_oovs: abstract对应article中出现的oov词汇列表
    """
    ids = []  # 存放最终的index结果
    unk_id = vocab.word_to_id('<UNK>')  # 未知词汇对应的index
    for w in abstract_words:
        i = vocab.word_to_id(w)  # 通过词典获取当前单词的index
        if i == unk_id:  # 如果发现oov词
            if w in article_oovs:  # 如果该词在对应article的oov词汇表中
                vocab_idx = vocab.size() + article_oovs.index(w)  # 计算该oov词汇在article扩展词汇表对应的位置
                ids.append(vocab_idx)  # 加入该词
            else:  # 如果连对应article的oov词汇表中都没有该单词，那真的就是一个unknown word了
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def example_generator(params, vocab):
    """
    训练数据或测试数据的生成器
    :param params: 参数字典
    :param vocab: 字典
    """
    if params["mode"] == "train":  # 训练模式，产生训练数据
        # 载入训练集的特征X和标签Y，注意这里不用numpy数组的数据格式，而是用训练集X和Y切割后的.csv文件
        ds_train_x = tf.data.TextLineDataset(train_x_seg_path)  # train_X_seg.csv
        ds_train_y = tf.data.TextLineDataset(train_y_seg_path)  # train_Y_seg.csv

        print(ds_train_x)

        # 合并训练数据中的X和Y为一个Dataset
        train_dataset = tf.data.Dataset.zip((ds_train_x, ds_train_y))
        # repeat()不加参数，代表数据无限地循环
        train_dataset = train_dataset.shuffle(params["batch_size"] * 2 + 1, reshuffle_each_iteration=True).repeat()

        for raw_record in train_dataset:  # 对于数据集中的每一条数据(包括articles和abstracts)

            # article的处理
            article = raw_record[0].numpy().decode("utf-8")  # 以'start 新能源 车 最大 短板 end'为例
            article_words = article.split()[:params['max_enc_len']]  # ['start', '新能源','车','最大','短板', 'end']

            enc_input = [vocab.word_to_id(w) for w in article_words]  # [7, 6080, 14, 1250, 14701, 8]

            enc_len = len(enc_input)  # encder输入的长度(例中为6)
            enc_mask = [1 for _ in range(enc_len)]  # [1, 1, 1, 1, 1, 1]，作为encoder的mask

            extended_enc_input, article_oovs = article_to_ids(article_words, vocab)  # 将一条articles转为编码器输入及OOV

            # abstract的处理
            abstract = raw_record[1].numpy().decode("utf-8")  # 以'start 在于 充电 还有 一个 续航  end'为例
            abstract_words = abstract.split()[
                             :params['max_dec_len']]  # ['start', '在于', '充电', '还有', '一个', '续航', '里程', 'end']

            dec_input = [vocab.word_to_id(w) for w in abstract_words][:-1]  # [7, 4980, 939, 41, 27, 4013, 815]，去掉end
            target = abstract_to_ids(abstract_words, vocab, article_oovs)[
                     1:]  # [4980, 939, 41, 27, 4013, 815, 8]，去掉start

            dec_len = len(target)  # 7
            dec_mask = [1 for _ in range(dec_len)]  # [1, 1, 1, 1, 1, 1, 1]

            assert len(enc_input) == len(extended_enc_input), "ERROR: your code has something wrong!"

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "extended_enc_input": extended_enc_input,
                "article_oovs": article_oovs,
                "dec_input": dec_input,
                "dec_target": target,
                "dec_len": dec_len,
                "article": article,
                "abstract": abstract,
                "dec_mask": dec_mask,
                "enc_mask": enc_mask
            }
            yield output

    else:  # 测试模式，产生测试数据
        test_dataset = tf.data.TextLineDataset(test_x_seg_path)  # test_X_seg.csv

        for raw_record in test_dataset:
            article = raw_record.numpy().decode("utf-8")  # 'start 新能源 车 最大 短板 end'
            article_words = article.split()[:params["max_enc_len"]]  # ['start', '新能源', '车', '最大', '短板', 'end']

            enc_input = [vocab.word_to_id(w) for w in article_words]  # [7, 6080, 14, 1250, 14701, 8]

            enc_len = len(enc_input)  # 6
            enc_mask = [1 for _ in range(enc_len)]  # [1, 1, 1, 1, 1, 1]

            extended_enc_input, article_oovs = article_to_ids(article_words, vocab)

            assert len(enc_input) == len(extended_enc_input), "ERROR: your code has something wrong!"

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "extended_enc_input": extended_enc_input,
                "article_oovs": article_oovs,
                "dec_input": [],
                "dec_target": [],
                "dec_len": params['max_dec_len'],
                "article": article,
                "abstract": '',
                "dec_mask": [],
                "enc_mask": enc_mask
            }

            yield output


def batch_generator(generator, params, vocab):
    output_types = {  # 输出数据类型字典
        "enc_len": tf.int32,
        "enc_input": tf.int32,
        "extended_enc_input": tf.int32,
        "article_oovs": tf.string,
        "dec_input": tf.int32,
        "dec_target": tf.int32,
        "dec_len": tf.int32,
        "article": tf.string,
        "abstract": tf.string,
        "dec_mask": tf.float32,
        "enc_mask": tf.float32}

    output_shapes = {  # 输出形状字典
        "enc_len": [],
        "enc_input": [None],
        "extended_enc_input": [None],
        "article_oovs": [None],
        "dec_input": [None],
        "dec_target": [None],
        "dec_len": [],
        "article": [],
        "abstract": [],
        "dec_mask": [None],
        "enc_mask": [None]}

    padded_shapes = {"enc_len": [],
                     "enc_input": [None],
                     "extended_enc_input": [None],
                     "article_oovs": [None],
                     "dec_input": [params['max_dec_len']],
                     "dec_target": [params['max_dec_len']],
                     "dec_len": [],
                     "article": [],
                     "abstract": [],
                     "dec_mask": [params['max_dec_len']],
                     "enc_mask": [None]}

    padding_values = {"enc_len": -1,
                      "enc_input": 0,
                      "extended_enc_input": 0,
                      "article_oovs": b'',
                      "dec_input": 0,
                      "dec_target": 0,
                      "dec_len": -1,
                      "article": b'',
                      "abstract": b'',
                      "dec_mask": 0.,
                      "enc_mask": 0.}

    # 从生成器中获取数据集
    dataset = tf.data.Dataset.from_generator(lambda: generator(params, vocab), output_types=output_types,
                                             output_shapes=output_shapes)
    # 注意，encoder和decoder中padding的value均为0
    dataset = dataset.padded_batch(params["batch_size"], padded_shapes=padded_shapes, padding_values=padding_values)

    def update(entry):
        # 重映射字典，输出分成2个字典一个是enc的输入，一个是dec的输入，只提取generator中有用的部分
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["extended_enc_input"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "enc_mask": entry["enc_mask"]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["dec_target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"],
                 "dec_mask": entry["dec_mask"]})

    dataset = dataset.map(update)  # 将字典键值对进行重映射
    return dataset


def batcher(vocab, params):
    dataset = batch_generator(example_generator, params, vocab)
    return dataset


if __name__ == "__main__":
    # config_gpu()  # GPU资源配置

    params = get_params()  # 获取参数
    vocab = Vocab(vocab_path)  # Vocab对象

    # 创建数据集，并获取一个batch
    dataset = batcher(vocab, params)
    print(dataset)

    batch = next(iter(dataset.take(1)))
    articles = batch[0]['article']
    abstracts = batch[1]['abstract']

    print(str(articles[0].numpy(), 'utf-8'))
    print(str(abstracts[0].numpy(), 'utf-8'))
