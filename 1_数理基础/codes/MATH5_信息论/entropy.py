# 实现信息论中的熵及交叉熵

import math

# 计算不同概率分布的熵
p1 = [0.1, 0.3, 0.6]
p2 = [0.33, 0.33, 0.34]
p3 = [0.0, 0.0, 1]


def entropy(p):
    ent = 0.0
    for i in range(len(p)):
        if p[i] != 0:
            ent += p[i] * math.log2(p[i])
    if ent != 0.0:
        return -ent
    else:
        return 0.0


print(entropy(p1))
print(entropy(p2))
print(entropy(p3))

# 交叉熵的实现(交叉熵广泛应用于机器学习中的多分类任务)
p_label = [1.0, 0.0, 0.0, 0.0, 0.0]
p_predict = [0.78, 0.11, 0.02, 0.06, 0.03]


def cross_entropy(p, q):
    ent = 0.0
    assert len(p) == len(q)
    for i in range(len(p)):
        if q[i] != 0:
            ent += p[i] * math.log2(q[i])
    if ent != 0.0:
        return -ent
    else:
        return 0.0


print(cross_entropy(p_label, p_predict))
