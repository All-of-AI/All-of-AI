import tensorflow as tf
import pandas as pd
import time
import os

from utils.config import test_raw_data_path, pgn_ckpt, result_save_path, vocab_path
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from pgn_model.batcher import Vocab
from pgn_model.batcher import batcher
from pgn_model.model import PGN
from pgn_model.test_helper import greedy_decode


def test(params):
    params['mode'] = 'test'

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(vocab_path)

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, pgn_ckpt, max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    # greeedy decode并保存结果
    results = greedy_predict_result(model, params, vocab, result_save_path)
    print('save result to :{}'.format(result_save_path))

    return results


def greedy_predict_result(model, params, vocab, result_save_path):
    dataset = batcher(vocab, params)
    # 预测结果
    results = greedy_decode(model, dataset, vocab, params)
    results = list(map(lambda x: x.replace(" ", ""), results))
    # 保存测试结果
    save_predict_result(results, result_save_path)
    return results


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(test_raw_data_path)
    # 填充结果
    test_df['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(get_result_file_name(), index=None, sep=',')


def get_result_file_name():
    """
    获取result file的名称(依据时间来命名)
    :return: 文件名称的名称
    """
    now = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = os.path.join(result_save_path, 'pgn_' + now + "_result.csv")
    return file_name


if __name__ == '__main__':
    # config_gpu() # GPU资源配置

    # 获得参数
    params = get_params()
    params["mode"] = "test"
    # 获得参数
    results = test(params)
