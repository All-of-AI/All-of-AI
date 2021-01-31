import tensorflow as tf
from seq2seq_model.model import Seq2Seq
from seq2seq_model.train_helper import train_model
from utils.config import vocab_path, seq2seq_ckpt
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from utils.load_data import load_vocab_from_txt


def train(params):
    # GPU资源配置
    # config_gpu()

    # 读取vocab训练
    vocab, _ = load_vocab_from_txt(vocab_path)

    # 构建模型
    print("Training mode")
    model = Seq2Seq(params)

    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, seq2seq_ckpt, max_to_keep=5)  # 最多设置5个检查点
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("模型从检查点中恢复: {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("无检查点，初始创建模型")

    # 训练模型
    print('开始训练模型')
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
