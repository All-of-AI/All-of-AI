# 正则表达式库re的使用示例

import re

# 将正则表达式编译成pattern对象
pattern = re.compile(r'hello.*\!')

# 使用pattern匹配文本，获得匹配结果，无法匹配时将返回None
match = pattern.match('hello, world! hello ! hello ?')

if match:
    # 使用Match获得分组信息
    print(match.group())

# re.match只匹配字符串的开始，若字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配
line = "Cats are smarter than dogs"

matchObj = re.match(r'dogs', line, re.M | re.I)  # match
if matchObj:
    print("match --> matchObj.group() : ", matchObj.group())
else:
    print("No match!!")

matchObj = re.search(r'dogs', line, re.M | re.I)  # search
if matchObj:
    print("search --> matchObj.group(): ", matchObj.group())
else:
    print("No match!!")

phone = "2004-959-559 # 这是一个电话号码"

# 删除注释
num = re.sub(r'#.*$', "", phone)
print("电话号码 : ", num)

# 移除非数字的内容
num = re.sub(r'\D', "", phone)
print("电话号码 : ", num)
