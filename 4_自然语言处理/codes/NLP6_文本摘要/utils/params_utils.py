import argparse


def get_params():
    """
    获取参数字典
    :return: 参数字典
    """
    parser = argparse.ArgumentParser()
    # 运行模型，为train或test
    parser.add_argument("--mode", default='train', help="Run mode", type=str)
    # 设定词典大小
    parser.add_argument("--vocab_size", default=15000 + 1, help="Size of vocab", type=int)
    # 编码器和解码器的最大序列长度
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=50, help="Decoder input max sequence length", type=int)
    # 一个训练批次的大小
    parser.add_argument("--batch_size", default=32, help="Batch size", type=int)
    # seq2seq训练轮数
    parser.add_argument("--seq2seq_train_epochs", default=20, help="Seq2seq model training epochs", type=int)
    # seq2seq已训练轮数，用于学习率衰减
    parser.add_argument("--trained_epoch", default=0, help="Seq2seq model trained epoch", type=int)
    # 集束搜索宽度
    parser.add_argument("--beam_size", default=3, help="Beam size for beam search decoding", type=int)
    # 词嵌入大小
    parser.add_argument("--embed_size", default=500, help="Words embeddings dimension", type=int)
    # 编码器、解码器以及attention的隐含层单元数
    parser.add_argument("--enc_units", default=512, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=512, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=20, help="Used to compute the attention weights", type=int)
    # 学习率
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    # coverage损失的权重
    parser.add_argument("--cov_loss_wt", default=1.0, help="Coverage loss weight", type=float)

    parser.add_argument("--steps_per_epoch", default=82871 // 32, help="How many steps in one spoch", type=int)
    args = parser.parse_args()
    params = vars(args)
    return params  # param是一个字典类型的变量，键为参数名，值为参数值
