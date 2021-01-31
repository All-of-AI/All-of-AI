import pandas as pd
import re

result_path = '../results/ernie_result.csv'

result = pd.read_csv(result_path, engine='python', encoding='utf-8')
predictions = result['Prediction']

for i in range(len(predictions)):
    # 删除一些无关紧要的词以及语气助词
    r = re.compile(r"车主说：|技师说：|\[语音\]|\[图片\]|呢|吧|哈|啊|啦|呕|嗯|吗|不客气")
    predictions[i] = r.sub("", predictions[i])
    # 将多个空格合并为一个空格
    r = re.compile(r" +")
    predictions[i] = r.sub("", predictions[i])
    if len(predictions[i]) <= 2:
        predictions[i] = '随时联系。'

# 重新保存结果
result['Prediction'] = predictions
# 提取ID和预测结果两列
result = result[['QID', 'Prediction']]

result.to_csv('../results/ernie_result_post_processed.csv', index=None, sep=',')
