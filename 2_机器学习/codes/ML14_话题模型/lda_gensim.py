# 使用gensim中的潜在狄利克雷分配对题目文本进行主题分析

import gensim
from gensim import corpora, models
import pandas as pd

df = pd.read_csv('data.txt', encoding='utf-8', sep='\t', quoting=3)
print(df.head)

df = df.dropna()
docs = df['segs']  # 选出文档部分，该部分为分词后的题干
print(docs.head)

doclist = docs.values
print(doclist[0])
print(doclist[1])
print(doclist[2])

texts = [[word for word in doc.split(' ') if len(word) > 1] for doc in doclist]  # 按空格切分，得到单词列表
print(texts[0])

# 词袋模型处理
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus[2])

# 开始训练LDA主题模型
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)  # num_topics为主题数
print(lda.print_topic(1, topn=20))
print(lda.print_topics(num_topics=20, num_words=20))

# 样本文档
bow_sample = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 3), (6, 1), (7, 3), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
              (13, 1), (14, 1), (15, 1), (16, 1), (17, 2), (18, 1), (19, 3), (20, 1), (21, 1), (22, 1), (23, 1),
              (24, 1), (25, 1), (26, 2), (27, 3), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1),
              (35, 1), (36, 1), (37, 1)]

# 获取样本文档的主题
print(lda.get_document_topics(bow_sample))
