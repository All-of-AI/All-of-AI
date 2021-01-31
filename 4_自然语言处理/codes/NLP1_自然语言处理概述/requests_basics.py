# 使用requests库配合bs4库实现网页内容的获取与解析

import requests
from bs4 import BeautifulSoup

# 使用requests库获取一个URL的HTML5内容
res = requests.get(url="http://tiku.21cnjy.com/tiku.php?mod=quest&channel=8&cid=1155&xd=2")
print(res.text)

# 使用BeautifulSoup库对HTML文本进行解析
soup = BeautifulSoup(res.text, 'html.parser')
target = soup.find(attrs={"class": "questions_col"})

li_list = target.find_all('li')

for item in li_list:
    print(item.text.replace(" ", "").replace('\t', ''))
