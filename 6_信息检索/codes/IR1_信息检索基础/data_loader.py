import pandas as pd


def load_data(processed_file_name, raw_file_name):
    df_processed = pd.read_csv(open(processed_file_name, encoding='utf8'))  # 先open，后read_csv的读取方式
    text_list = df_processed['content'].tolist()

    inverted_index = {}
    vocab = {}

    # 将text_list中每个文档分为词汇列表
    for i in range(len(text_list)):
        text_list[i] = text_list[i].split(' ')

    word_count = 0
    # 构造词典和倒排索引
    for i in range(len(text_list)):
        for item in text_list[i]:

            if item not in vocab.keys():
                vocab[item] = word_count
                word_count += 1

            if item not in inverted_index.keys():
                inverted_index[item] = []  # 初始化
                inverted_index[item].append(i)  # 加入文档id
            else:
                if inverted_index[item][-1] != i:  # 避免重复
                    inverted_index[item].append(i)

    # 导入原始题目文档
    df_raw = pd.read_csv(open(raw_file_name, encoding='utf8'), header=None)  # 先open，后read_csv的读取方式
    df_raw.columns = ['label', 'content']
    text_list = df_raw['content'].tolist()
    docs = []
    for doc in text_list:
        docs.append("".join(doc))

    # 返回倒排索引、词典和文档列表
    return inverted_index, vocab, docs


if __name__ == '__main__':
    # 一次性完成文档列表、倒排索引及词典的读取
    processed_data_path = 'data/data_processed.csv'  # 数据已完成预处理及分词
    raw_data_path = 'data/data_raw.csv'  # 原始数据

    inverted_index, vocab, docs = load_data(processed_data_path, raw_data_path)

    print(inverted_index)
    # print(vocab)
    # print(docs[:5])
