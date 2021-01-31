# requests和beautifulsoup4模块的联合使用

import requests
from bs4 import BeautifulSoup
import uuid

# 爬取汽车之家网站，获取每条新闻的标题，标题链接和图片
reponse = requests.get(url="https://www.autohome.com.cn/news/")
reponse.encoding = reponse.apparent_encoding  # 获取文本原来编码，使两者编码一致才能正确显示

soup = BeautifulSoup(reponse.text, 'html.parser')  # 使用的是html解析，一般使用lxml解析更好
target = soup.find(id="auto-channel-lazyload-article")  # find根据属性去获取对象，id,attr,tag...自定义属性
li_list = target.find_all('li')  # 列表形式
for li in li_list:
    a_tag = li.find('a')
    if a_tag:
        href = a_tag.attrs.get("href")  # 属性是字典形式，使用get获取指定属性
        title = a_tag.find("h3").text  # find获取的是对象含有标签，获取text
        img_src = "http:" + a_tag.find("img").attrs.get('src')
        print(href)
        print(title)
        print(img_src)
        img_reponse = requests.get(url=img_src)
        file_name = str(uuid.uuid4()) + '.jpg'  # 设置一个不重复的图片名
        with open(file_name, 'wb') as fp:
            fp.write(img_reponse.content)
