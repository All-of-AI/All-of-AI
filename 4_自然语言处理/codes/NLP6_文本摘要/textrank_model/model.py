import pandas as pd
import re
from textrank4zh import TextRank4Sentence

test_raw_data_path = '../data/AutoMaster_TestSet.csv'  # 测试语料路径，TextRank为无监督学习
wv_model_path = '../data/wv/word2vec.model'

# 读取数据
test_df = pd.read_csv(test_raw_data_path, engine='python', encoding='utf-8')  # 必须utf-8
texts = test_df['Dialogue'].tolist()
print('预处理前的第一条句子：', texts[1])


def clean_sentence(sentence):
    """
    句子预处理
    :param sentence: 待处理的字符串
    :return: 预处理后的字符串
    """
    # 1. 将sentence按照'|'分句，并只提取技师的话
    sub_jishi = []
    sub = sentence.split('|')  # 按照'|'字符将车主和用户的对话分离

    for i in range(len(sub)):  # 遍历每个子句
        if not sub[i].endswith('。'):  # 如果不是以句号结尾，增加一个句号
            sub[i] += '。'
        if sub[i].startswith('技师'):  # 只使用技师说的句子
            sub_jishi.append(sub[i])

    sentence = ''.join(sub_jishi)

    # 2. 删除1. 2. 3. 这些标题
    r = re.compile("\D(\d\.)\D")
    sentence = r.sub("", sentence)

    # 3. 删除带括号的 进口 海外
    r = re.compile(r"[(（]进口[)）]|\(海外\)")
    sentence = r.sub("", sentence)

    # 4. 删除除了汉字数字字母和，！？。.- 以外的字符
    r = re.compile("[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")
    # 半角变为全角
    sentence = sentence.replace(",", "，")
    sentence = sentence.replace("!", "！")
    sentence = sentence.replace("?", "？")
    # 问号叹号变为句号
    sentence = sentence.replace("？", "。")
    sentence = sentence.replace("！", "。")
    sentence = r.sub("", sentence)

    # 5. 删除一些无关紧要的词以及语气助词
    r = re.compile(r"车主说|技师说|语音|图片|呢|吧|哈|啊|啦|呕|嗯")
    sentence = r.sub("", sentence)

    # 6. 把，。/。。/。，替换为一个。(不执行这一步，ROUGE反而会有提升)
    # sentence = sentence.replace("，。", "。")
    # sentence = sentence.replace("。。", "。")
    # sentence = sentence.replace("。，", "。")

    # 7. 删除句子开头的逗号
    if sentence.startswith('，'):
        sentence = sentence[1:]

    # 8. 删除废话
    sub_sentences = re.split(r'[。]', sentence)
    new_sent = []
    for sub_sent in sub_sentences:  # 对于每个以句号结尾的子句
        segs = re.split(r'[，]', sub_sent)
        new_segs = []
        for seg in segs:
            if len(seg) > 0 and re.search('祝您|祝你|客气|汽车大师|吗|技师|追问|随时联系|帮到|请问|谢', seg) is None:
                new_segs.append(seg)
        new_sub_sent = '，'.join(new_segs)
        new_sub_sent += '。'
        new_sent.append(new_sub_sent)
    sentence = ''.join(new_sent)
    return sentence


# 数据预处理
for i in range(len(texts)):  # 20000
    texts[i] = clean_sentence(texts[i])
print('预处理后的第一条句子：', texts[1])

# 进行预测
results = []  # results保存所有20000个摘要结果
tr4s = TextRank4Sentence()  # TextRank4Sentence对象

for i in range(len(texts)):
    text = texts[i]
    text_divide = re.split('[。]', text)

    sort_dict = {j: text_divide[j] for j in range(len(text_divide))}  # 句子按照原顺序排序的字典(次序: 句子)
    reverse_sort_dict = {text_divide[j]: j for j in range(len(text_divide))}  # 反向排序字典(句子: 次序)

    # 利用tr4s完成摘要
    tr4s.analyze(text=text, lower=True, source='all_filters')
    result = tr4s.get_key_sentences(num=3, sentence_min_len=3)  # 选取长度大于等于3的最重要的3条句子
    result_list = []
    for item in result:
        result_list.append(item.sentence)

    for j in range(len(result_list)):
        result_list[j] = reverse_sort_dict[result_list[j]]  # 变成顺序下标

    result_list.sort()  # 对顺序下标进行排序

    for j in range(len(result_list)):
        result_list[j] = sort_dict[result_list[j]]  # 利用词典恢复成句子

    if len(result_list) == 0:
        result_list = '随时联系'
    else:
        result_list = '。'.join(result_list)
        result_list += '。'

    results.append(result_list)

    # 间隔100次打印结果
    if (i + 1) % 100 == 0:
        print(i + 1, result_list)

# 保存结果
test_df['Prediction'] = results
# 提取ID和预测结果两列
test_df = test_df[['QID', 'Prediction']]
# 将空行置换为随时联系
test_df = test_df.fillna('随时联系。')
test_df.to_csv('../results/textrank_result.csv', index=None, sep=',')
