import tensorflow as tf

from pgn_model.model import PGN
from pgn_model.train_helper import train_model
from pgn_model.batcher import Vocab
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from utils.config import pgn_ckpt, vocab_path
import numpy as np


def train(params):
    # config_gpu()  # GPU资源配置
    params['mode'] = 'train'
    vocab = Vocab(vocab_path)  # 读取vocab
    params['learning_rate'] *= np.power(0.95, params["trained_epoch"])  # 学习率衰减

    # 构建模型
    print("构建模型")
    model = PGN(params)

    # 获取保存管理者并尝试恢复模型，若没有检查点，则创建新的模型进行训练
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, pgn_ckpt, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("模型从检查点中恢复: {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("无检查点，初始创建模型")

    # 训练模型
    print("开始训练模型")
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
