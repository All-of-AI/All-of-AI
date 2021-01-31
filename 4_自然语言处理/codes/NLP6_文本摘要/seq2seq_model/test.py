import tensorflow as tf
import pandas as pd
import time

from seq2seq_model.model import Seq2Seq
from seq2seq_model.test_helper import greedy_decode
from utils.config import *
from utils.load_data import load_test_dataset
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from utils.load_data import load_vocab_from_txt


def test(params):
    # config_gpu()  # GPU资源配置

    print("Test mode")
    vocab, reverse_vocab = load_vocab_from_txt(vocab_path)

    model = Seq2Seq(params)

    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, seq2seq_ckpt, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("模型从以下检查点中恢复: {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("模型无检查点，直接创建")

    # 读取测试数据
    test_X = load_test_dataset()

    # 使用greedy decode方式进行预测
    results = greedy_decode(model, test_X, vocab, reverse_vocab, params)
    results = list(map(lambda x: x.replace(" ", ""), results))  # 去掉预测结果之间的空格
    save_predict_result(results)
    return results


def save_predict_result(results):
    # 读取原始文件(使用其QID列，合并新增的Prediction后再保存)
    test_df = pd.read_csv(test_raw_data_path)
    # 填充结果
    test_df['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果，这里自动生成一个结果名
    test_df.to_csv(get_result_filename(), index=None, sep=',')


def get_result_filename():
    """
    获取result file的名称(依据时间来命名)
    :return: 文件名称的名称
    """
    now = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = os.path.join(result_save_path, 'seq2seq_' + now + '_result.csv')
    return file_name


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 测试模型
    results = test(params)
