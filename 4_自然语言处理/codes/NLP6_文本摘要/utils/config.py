import os
import pathlib

# 项目根路径
root_path = pathlib.Path(os.path.abspath(__file__)).parent.parent  # config.py绝对路径的父路径(utils/)的父路径

# 训练原始数据.csv路径
train_raw_data_path = os.path.join(root_path, 'data', 'AutoMaster_TrainSet.csv')
# 测试原始数据.csv路径
test_raw_data_path = os.path.join(root_path, 'data', 'AutoMaster_TestSet.csv')
# 停用词.txt路径
stop_word_path = os.path.join(root_path, 'data', 'stopwords/stopwords.txt')

# 自定义切词表.txt路径
user_dict_path = os.path.join(root_path, 'data', 'user_dict.txt')

# 预处理+切分后的训练测试数据路径
train_seg_path = os.path.join(root_path, 'data', 'train_seg_data.csv')
test_seg_path = os.path.join(root_path, 'data', 'test_seg_data.csv')

# 预处理+切分后X和Y分离的训练测试数据路径
train_x_seg_path = os.path.join(root_path, 'data', 'train_X_seg.csv')
train_y_seg_path = os.path.join(root_path, 'data', 'train_Y_seg.csv')
test_x_seg_path = os.path.join(root_path, 'data', 'test_X_seg.csv')

# numpy转换为数字后最终使用的的数据路径
train_x_path = os.path.join(root_path, 'data', 'train_X.npy')
train_y_path = os.path.join(root_path, 'data', 'train_Y.npy')
test_x_path = os.path.join(root_path, 'data', 'test_X.npy')

# 正向词典和反向词典路径
vocab_path = os.path.join(root_path, 'data', 'vocab.txt')
inverse_vocab_path = os.path.join(root_path, 'data', 'inverse_vocab.txt')

# 模型检查点路径
pgn_ckpt = os.path.join(root_path, 'data', 'checkpoints', 'pgn_checkpoints')
seq2seq_ckpt = os.path.join(root_path, 'data', 'checkpoints', 'seq2seq_checkpoints')

# 测试集结果保存路径
result_save_path = os.path.join(root_path, 'results')
