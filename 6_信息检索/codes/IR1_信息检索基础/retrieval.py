from data_loader import load_data


def union(list_a, list_b):
    """
    将有序列表list_a和list_b进行合并操作，返回二者的公有元素
    """
    res = []
    i = j = 0
    while i < len(list_a) and j < len(list_b):
        if list_a[i] < list_b[j]:
            i += 1
        elif list_a[i] > list_b[j]:
            j += 1
        else:  # list_a[i] == list_b[j]
            res.append(list_a[i])
            i += 1
            j += 1
    return res


if __name__ == '__main__':
    processed_data_path = 'data/data_processed.csv'  # 数据已完成预处理及分词
    raw_data_path = 'data/data_raw.csv'  # 原始数据
    inverted_index, vocab, docs = load_data(processed_data_path, raw_data_path)

    input_text = input('请输入搜索关键词，按空格隔开: ')
    # 支持AND逻辑布尔运算
    input_words = input_text.split(' ')

    results = []
    for word in input_words:
        if word in inverted_index.keys():
            results.append(inverted_index[word])

    if len(results) > 1:
        for i in range(1, len(results)):
            results[0] = union(results[0], results[i])  # 令results[0]为最终检索结果

    if len(results) == 0:
        print('未检索到相关文档')
    else:
        print('共检索到{}条结果，如下：'.format(len(results[0])))

        for i in range(len(results[0])):
            print('检索结果{}: '.format(i + 1))
            print(docs[results[0][i]])
            print('\n')
