import tensorflow as tf

# 选用交叉熵损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def log_loss(target, pred, dec_mask):
    """
    计算log loss
    :param target: shape (batch_size, dec_len)
    :param pred: shape (batch_size, dec_len, vocab_size)
    :param dec_mask: shape (batch_size, dec_len)
    :return: log loss
    """
    loss = loss_object(target, pred)
    loss *= dec_mask
    loss = tf.reduce_mean(loss)
    return loss


def coverage_loss(attentions, coverages, dec_mask):
    """
    计算coverage loss，根据原文公式12
    :param attentions: shape (batch_size, dec_len, enc_len)
    :param coverages: shape (batch_size, dec_len, enc_len)
    :param dec_mask: shape (batch_size, dec_len)
    :return: cov_loss
    """
    # cov_loss (batch_size, dec_len, enc_len)
    cov_loss = tf.minimum(attentions, coverages)
    # mask
    cov_loss = tf.expand_dims(dec_mask, -1) * cov_loss
    # 对enc_len的维度求和
    cov_loss = tf.reduce_sum(cov_loss, axis=2)  # 对第二个维度求和后的shape: (batch_size, dec_len)
    cov_loss = tf.reduce_mean(cov_loss)  # reduce_mean后就是一个数了，因为其默认对所有维度求mean
    return cov_loss


def calc_loss(target, pred, dec_mask, attentions, coverages, cov_loss_weight=1, use_coverage=False):
    if use_coverage:
        log_loss_value = log_loss(target, pred, dec_mask)
        cov_loss_value = coverage_loss(attentions, coverages, dec_mask)
        # 使用coverage，返回log_loss+λ*cov_loss，以及分别的log_loss和cov_loss
        return log_loss_value + cov_loss_weight * cov_loss_value, log_loss_value, cov_loss_value
    else:
        # 不使用coverage，直接返回log_loss
        return log_loss(target, pred, dec_mask), log_loss(target, pred, dec_mask), 0
