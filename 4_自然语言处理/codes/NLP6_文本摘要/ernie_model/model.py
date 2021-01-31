import pandas as pd
import jieba
import numpy as np
import scipy
import multiprocessing as mp
from paddle import fluid
import paddlehub as hub
import paddle
import codecs
import os
import csv
from scipy.spatial import distance
from collections import namedtuple
from paddlehub.reader.tokenization import load_vocab
from paddlehub.dataset import InputExample, BaseDataset
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

from utils.config import train_raw_data_path, test_raw_data_path

train = pd.read_csv(train_raw_data_path, engine='python', encoding='utf-8').fillna(' ')
test = pd.read_csv(test_raw_data_path, engine='python', encoding='utf-8').fillna(' ')

print(train.head(2))

# 提取对话
dialogues = []
for dia in train['Dialogue']:
    dialogues.append(dia.split('|'))

print('对话数量: ', len(dialogues))  # 82943
print(dialogues[10])

# 将车主和技师的话分离
dialogues_pair = []

for dia in dialogues:
    line = []
    tmp = []
    b = True
    for d in dia:
        if d[:2] == '技师':
            tmp.append(d)
            b = True
        else:
            if b is True:
                line.append(list(tmp))
                tmp = [d]
            else:
                tmp.append(d)
            b = False
    if b is True:
        line.append(list(tmp))
    dialogues_pair.append(list(line))

print('对话对数量: ', len(dialogues_pair))  # 82943
print(dialogues_pair[10])


# 切词
def seg_line(line):
    tokens = jieba.cut(line, cut_all=False)
    return " ".join(tokens)


dialogues_pair_cut = []
for dia in dialogues_pair:
    p = []
    for dd in dia:
        line = ''
        for d in dd:
            line += ' '
            line += seg_line(d)
        p.append(line)
    dialogues_pair_cut.append(list(p))

print(dialogues_pair_cut[10])

# 处理回复
reports = list(train['Report'].values)

reports_cut = []
for r in reports:
    reports_cut.append(seg_line(r))

print(reports_cut[10])


# 将一段文本转化为id序列
def convert_tokens_to_ids(vocab, text):
    wids = []
    tokens = text.split(" ")
    for token in tokens:
        wid = vocab.get(token, None)
        if not wid:
            wid = vocab["unknown"]
        wids.append(wid)
    return wids


module = hub.Module(name="word2vec_skipgram")  # 导入词向量模型(hub install word2vec_skipgram==1.0.0)，注意版本
inputs, outputs, program = module.context(trainable=False)  # 词向量模型上下文
vocab = load_vocab(module.get_vocab_path())  # 导入词典

word_ids = inputs["word_ids"]
embedding = outputs["word_embs"]

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[word_ids], place=place)


# 文本相似度计算
def cal_sim(a, b):
    text_a = convert_tokens_to_ids(vocab, a)
    text_b = convert_tokens_to_ids(vocab, b)

    vecs_a, = exe.run(program, feed=feeder.feed([[text_a]]), fetch_list=[embedding.name], return_numpy=False)
    vecs_a = np.array(vecs_a)
    vecs_b, = exe.run(program, feed=feeder.feed([[text_b]]), fetch_list=[embedding.name], return_numpy=False)
    vecs_b = np.array(vecs_b)

    sent_emb_a = np.sum(vecs_a, axis=0)
    sent_emb_b = np.sum(vecs_b, axis=0)
    cos_sim = 1 - distance.cosine(sent_emb_a, sent_emb_b)
    return cos_sim


def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]


def process(dat):
    dialogues_pair_cut_chunk, reports_cut_chunk = dat[0], dat[1]
    dialogues_scores = []
    for i in range(len(dialogues_pair_cut_chunk)):
        dia = dialogues_pair_cut_chunk[i]
        rep = reports_cut_chunk[i]
        scores = []
        for d in dia:
            sim = cal_sim(d, rep)
            scores.append(sim)
        dialogues_scores.append(list(scores))
    return dialogues_scores


dia_chunk_list = split_df(dialogues_pair_cut, 100)
rep_chunk_list = split_df(reports_cut, 100)

chunk_list = [(dia_chunk_list[i], rep_chunk_list[i]) for i in range(len(dia_chunk_list))]

with mp.Pool() as pool:
    ret = pool.map(process, chunk_list)
    print(len(ret))

dialogues_scores = []
for r in ret:
    dialogues_scores += r

print(len(dialogues_scores))

trainset = []
for i1, row in train.iterrows():
    qid = row['QID']
    que = row['Question']
    dp = dialogues_pair[i1]
    scores = dialogues_scores[i1]
    tmp_list = sorted(scores)
    tmps = (tmp_list[-2] if len(tmp_list) > 1 else tmp_list[-1])
    for i2, d in enumerate(dp):
        s = scores[i2]
        s = (0 if s < tmps else 1)
        d = ' '.join(d)
        trainset.append([qid, que, d, s])

trainset = pd.DataFrame(trainset, columns=['qid', 'text_a', 'text_b', 'label'])
trainset['label'] = (trainset['label'] - trainset['label'].min()) / (trainset['label'].max() - trainset['label'].min())

train, dev = trainset[:int(len(trainset) * 0.9)], trainset[int(len(trainset) * 0.9):]
print(len(train), len(dev))

train['text_a'] = train['text_a'].apply(lambda x: x.replace('\t', ' '))
train['text_b'] = train['text_b'].apply(lambda x: x.replace('\t', ' '))
dev['text_a'] = dev['text_a'].apply(lambda x: x.replace('\t', ' '))
dev['text_b'] = dev['text_b'].apply(lambda x: x.replace('\t', ' '))

train['label'] = train['label'].astype(int)
dev['label'] = dev['label'].astype(int)

train.to_csv('./train.tsv', index=False, sep='\t')
dev.to_csv('./dev.tsv', index=False, sep='\t')

# 准备测试数据
test_dialogues = []
for dia in test['Dialogue']:
    test_dialogues.append(dia.split('|'))

test_dialogues_pair = []

for dia in test_dialogues:
    line = []
    tmp = []
    b = True
    for d in dia:
        if d[:2] == '技师':
            tmp.append(d)
            b = True
        else:
            if b is True:
                line.append(list(tmp))
                tmp = [d]
            else:
                tmp.append(d)
            b = False
    if b is True:
        line.append(list(tmp))
    test_dialogues_pair.append(list(line))

testset = []
for i1, row in test.iterrows():
    qid = row['QID']
    que = row['Question']
    dp = test_dialogues_pair[i1]
    for i2, d in enumerate(dp):
        d = ' '.join(d)
        testset.append([qid, que, d, 0])

testset = pd.DataFrame(testset, columns=['qid', 'text_a', 'text_b', 'label'])

testset['text_a'] = testset['text_a'].apply(lambda x: x.replace('\t', ' '))
testset['text_b'] = testset['text_b'].apply(lambda x: x.replace('\t', ' '))

testset['label'] = testset['label'].astype(int)

testset.to_csv('./test.tsv', index=False, sep='\t')


# 定义适用于paddle hub模型的输入数据格式(BaseDataset)
class MyDataset(BaseDataset):
    def __init__(self):
        self.dataset_dir = './'
        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.train_examples = self._read_tsv(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.dev_examples = self._read_tsv(self.dev_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.test_examples = self._read_tsv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[3], text_a=line[1], text_b=line[2])
                seq_id += 1
                examples.append(example)

            return examples


# 加载ERNIE模型及其上下文
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(trainable=True, max_seq_len=200)

dataset = MyDataset()
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=200)

strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_proportion=0.0,
    lr_scheduler="linear_decay",
)

config = hub.RunConfig(
    use_cuda=True,
    log_interval=100,
    eval_interval=10000,
    use_data_parallel=True,
    use_pyreader=True,
    num_epoch=5,
    batch_size=32,
    checkpoint_dir='ckpt_ernie',
    strategy=strategy)

pooled_output = outputs["pooled_output"]

# feed_list的Tensor顺序不可以调整
feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config)

cls_task.finetune_and_eval()

p = cls_task.eval()
print(p)

# 进行测试
print(testset.head())

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

config = hub.RunConfig(
    use_data_parallel=False,
    use_pyreader=False,
    use_cuda=True,
    batch_size=32,
    enable_memory_optim=False,
    checkpoint_dir='ckpt_ernie',
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

cls_task = hub.TextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config)

test_data = [[d.text_a, d.text_b] for d in dataset.get_test_examples()]

run_states = cls_task.predict(test_data)

results = [run_state.run_results for run_state in run_states]

predicts = []
for batch_result in results:
    # 预测类别取最大分类概率值
    predict = np.argmax(batch_result[0], axis=1)
    print(predict)
    for item in predict:
        predicts.append(item)

testset['label'] = predicts
print(testset.head())

maxp_idx = testset.groupby('qid')['label'].apply(lambda x: np.argmax(x)).reset_index()

maxp_idx.columns = ['qid', 'label_idx']
res_df = testset.merge(maxp_idx, 'left', 'qid')

res_df2 = res_df[res_df.index == res_df.label_idx]

res_df3 = res_df2[['qid', 'text_b']]
res_df3.columns = ['QID', 'Prediction']
res_df3.to_csv('ernie_result.csv', index=False)
