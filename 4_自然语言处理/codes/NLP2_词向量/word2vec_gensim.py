# 使用gensim库中的word2vec工具训练词向量，语料库选用某IT招聘网站的信息

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

f_obj = open('data_split.txt', 'r', encoding='utf-8')  # 语料库中的单词被空格分隔
data = f_obj.read()
data = data.split('\n')
print(len(data))  # 85399条数据

# 按空格分离单词串为列表
for i in range(len(data)):
    data[i] = data[i][:-1].split(' ')

# 构建word2vec模型
word2vec_model = Word2Vec(data, size=300, min_count=10, iter=5)  # 300维

# 获取正向词典和反向词典
vocab = {word: index for index, word in enumerate(word2vec_model.wv.index2word)}
reverse_vocab = {index: word for index, word in enumerate(word2vec_model.wv.index2word)}
print(vocab)
print(reverse_vocab)

# 获取词嵌入矩阵
embedding_matrix = word2vec_model.wv.vectors
print('shape of embedded matrix: ', embedding_matrix.shape)  # 词典大小：15368

# 寻找某个单词在向量空间中最近的10个单词
print('most similar top 10 words with "python": ', word2vec_model.wv.most_similar(['python'], topn=10))
print('most similar top 10 words with "产品": ', word2vec_model.wv.most_similar(['产品'], topn=10))
print('most similar top 10 words with "良好": ', word2vec_model.wv.most_similar(['良好'], topn=10))

SAVE_MODEL_PATH = 'word2vec.model'

# 保存模型
word2vec_model.save(SAVE_MODEL_PATH)

# 加载模型
word2vec_model_load = Word2Vec.load(SAVE_MODEL_PATH)
