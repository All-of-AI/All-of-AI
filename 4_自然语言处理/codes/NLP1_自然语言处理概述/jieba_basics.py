# jieba库的分词、词性标注与关键词提取操作示例

import jieba

text = '2013年政府工作报告提出，缩小收入分配差距，使发展成果更多更公平地惠及全体人民，这与以往政府提出的“尽快扭转收入差' \
       '距扩大趋势”相比，有了一个更高的视角。下列对收入分配差距认识正确的是：①收入分配差距的存在违背了社会主义的本质要求；' \
       '②收入分配差距过大，违背了公平原则，不利于社会稳定；③适度的收入分配差距有利于激发劳动者的生产积极性；④消除收入差距' \
       '是社会土义市场经济的根本目标'

seg_result1 = jieba.cut(text, cut_all=True)  # 全切分
print('cut_all true：', ' '.join(seg_result1))

seg_result2 = jieba.cut(text, cut_all=False)
print('cut_all false：', ' '.join(seg_result2))

seg_result3 = jieba.cut(text)
print('cut_all default：', ' '.join(seg_result3))

seg_result4 = jieba.cut_for_search(text)
print('cut_for_search：', ' '.join(seg_result4))

# 使用jieba.load_userdict(file_name)可以加载用户词表

# 使用jieba.posseg进行词性标注
import jieba.posseg as pseg

words = pseg.cut(text)
for word, flag in words:
    print('%s %s' % (word, flag))

# 使用jieba.analyse进行关键词提取
import jieba.analyse as analyse

print(" ".join(analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=())))
print(" ".join(analyse.textrank(text, topK=20, withWeight=False, allowPOS=('ns', 'n'))))
