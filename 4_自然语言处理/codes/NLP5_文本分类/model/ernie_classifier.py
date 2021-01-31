import os
import pandas as pd
import paddlehub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from collections import namedtuple
import codecs
import os
import csv
import numpy as np

from paddlehub.dataset.dataset import InputExample, BaseDataset
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

# 读取数据
DATA_OUTPUT_DIR = '../data/'
data_path = os.path.join(DATA_OUTPUT_DIR, 'baidu_95.csv')
df = pd.read_csv(data_path, header=None, names=["labels", "content"], dtype=str)

print(df.head(3))

# 数据预处理
df['labels'] = df['labels'].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])
y = [' '.join([str(j) for j in i]) for i in y.tolist()]
df['labels'] = y

classes_df = pd.DataFrame(mlb.classes_)

print(classes_df.head(10))

print(df.head())

# 训练数据集划分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # 80%的数据集用作训练
# 测试和验证集划分
test_df, dev_df = train_test_split(test_df, test_size=0.5, random_state=42)  # 1:1除去训练集用作test和dev

# 保存数据
train_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'train.tsv'), index=None, header=None)
dev_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'dev.tsv'), index=None, header=None)
test_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'test.tsv'), index=None, header=None)

# 保存类别标签
classes_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'classes.csv'), index=None, header=None)


# 数据加载
class Baidu95(BaseDataset):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for opinion mining)
    """
    def __init__(self):
        self.dataset_dir = DATA_OUTPUT_DIR
        if not os.path.exists(self.dataset_dir):
            logger.info("Dataset not exists.".format(self.dataset_dir))
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

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
        return pd.read_csv(os.path.join(DATA_OUTPUT_DIR, 'classes.csv'), names=['labels'])['labels'].tolist()

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=[int(i) for i in line[0].split(' ')], text_a=line[1])
                seq_id += 1
                examples.append(example)

            return examples


# 更换name参数即可无缝切换BERT中文模型, 代码示例如下
max_seq_len = 256
module = hub.Module(name="ernie_tiny")  # ERNIE_tiny
# module = hub.Module(name="bert_chinese_L-12_H-768_A-12")  # BERT
inputs, outputs, program = module.context(trainable=True, max_seq_len=max_seq_len)

# 构建Reader
dataset = Baidu95()
reader = hub.reader.MultiLabelClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=max_seq_len,
    use_task_id=False)
metrics_choices = ['acc', 'f1']

# 优化器设置
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_proportion=0.0,
    lr_scheduler="linear_decay",
)

config = hub.RunConfig(use_cuda=True, num_epoch=5, batch_size=32, strategy=strategy)

# 运行模型
# Define a classfication finetune task by PaddleHub's API
pooled_output = outputs["pooled_output"]
feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name
]

# 定义多标签分类任务
multi_label_cls_task = hub.MultiLabelClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=dataset.num_labels,
    config=config)

# Fine-tune and evaluate by PaddleHub's API
# will finish training, evaluation, testing, save model automatically
multi_label_cls_task.finetune_and_eval()

# 预测
data = [[d.text_a, d.text_b] for d in dataset.get_test_examples()]
# 预测标签
test_label = np.array([d.label for d in dataset.get_test_examples()])
# 预测
run_states = multi_label_cls_task.predict(data)


def inverse_predict_array(batch_result):
    return np.argmax(batch_result, axis=2).T


results = [run_state.run_results for run_state in run_states]
predict_label = np.concatenate([inverse_predict_array(batch_result) for batch_result in results])

print('f1 micro: {}'.format(f1_score(test_label, predict_label, average='micro')))
print('f1 samples: {}'.format(f1_score(test_label, predict_label, average='samples')))
print('f1 macro: {}'.format(f1_score(test_label, predict_label, average='macro')))

print(mlb.inverse_transform(predict_label)[2:5])
print(mlb.inverse_transform(test_label)[2:5])
