## 自然语言处理8 机器阅读理解

### 8.1 概述

#### 8.1.1 MRC的形式化定义

给定上下文$C$以及问题$Q$，**机器阅读理解(machine reading comprehension, MRC)**任务要求模型给出问题$Q$正确的回答$A$，该过程通过学习函数$\mathcal F$得到，即$A=\mathcal F(C,Q)$[1]。

#### 8.1.2 MRC中的四大任务

机器阅读理解的四大任务分别为：**完形填空(cloze test)、多项选择(multiple choice)、答案抽取(span extraction)以及自由问答(free answering)**。

<img src="images/image-20200610151552516.png" style="zoom:30%;" />

四大任务的**常用数据集**如下：

(1) 完形填空：**CNN & Daily Mail**，The Children's Book Test，LAMBADA。

<img src="images/image-20200610152037046.png" style="zoom:50%;" />

(2) 多项选择：MC Test，RACE。

<img src="images/image-20200610152107786.png" style="zoom:50%;" />

(3) 答案抽取：**SQuAD**，**NewsQA**，TriviaQA。

<img src="images/image-20200610152142781.png" style="zoom:50%;" />

(4) 自由问答：**bAbI**，MS MARCO，SearchQA，**DuReader**[http://ai.baidu.com/broad/download]。

<img src="images/image-20200610152327311.png" style="zoom:50%;" />

#### 8.1.3 MRC的四大基础模块

MRC任务有以下四大基础模块：

![image-20200610153403730](images/image-20200610153403730.png)

(1) **Embeddings**：该模块将context和question嵌入到向量空间中，使用包含语义信息的向量来表示单词、句子以及段落的含义。常用的方法有one-hot、word2vec、**预训练语言模型(ELMo、GPT、BERT)**等，还可以**融合其他特征**，例如字符嵌入(character embedidng)、词性、命名实体等。

(2) **Feature Extraction**：该模块用于提取context和question更多上下文相关的特征。常用模型有RNN、CNN、Transformer等。

(3) **Context-Question Interaction**：该模块通过将context和question进行特征关联来产生好的答案。常用模型有单向注意力机制、**双向注意力机制(bidirectional attention)**、one-hop交互和**multi-hop交互**。该模块被认为是机器阅读理解领域最重要的模块。

(4) **Answer Prediction**：该模块生成答案。根据四大任务，产生答案的方式也对应有**word predictor、option selector、span extractor和answer generator**四种。

#### 8.1.4 MRC的主要评测方法和指标

(1) accuracy：主要用于**完形填空**和**多项选择**任务。当共有$m$个任务，有$n$道回答正确，则accuracy为：
$$
\text{Accuracy} =\frac{n}{m}
$$
(2) F1-score：主要用于**答案抽取**任务。在答案抽取任务中，**候选答案(产生的答案)**和**参考答案(实际的答案)**都被当做**a bag of tokens**，可以给出混淆矩阵：

<img src="images/image-20200610164905946.png" style="zoom:40%;" />

精准率、召回率以及F1-score的计算方式分别如下：
$$
\begin{aligned}
\text{precision}&=\frac{\text{TP}}{\text{TP}+\text{FP}}\\
\text{recall}&=\frac{\text{TP}}{\text{TP}+\text{FN}}\\
\text{F}_1&=\frac{2 \times P \times R}{P+R}
\end{aligned}
$$
(3) ROUGE-L：主要用于**自由问答**任务，pyrouge工具可以很容易地实现ROUGE值的计算。


$$
\begin{aligned}
R_{lcs}&=\frac{LCS(X, Y)}{m} \\
P_{lcs}&=\frac{LCS(X, Y)}{n} \\
F_{lcs}&=\frac{(1+\beta)^{2} R_{lcs} P_{lcs}}{R_{lcs}+\beta^{2} P_{lcs}}
\end{aligned}
$$


(4) BLEU：主要用于**自由问答**任务。BLEU的计算主要基于如下指标：

$$
P_{n}(C, R)=\frac{\sum_{i} \sum_{k} \min (h_{k}(c_{i}), \max (h_{k}(r_{i})))}{\sum_{i} \sum_{k} h_{k}(c_{i})}
$$
其中$h_k(c_i)$指**候选答案**$c_i$中出现的第$k$个**n-gram的数量**，$h_k(r_i)$指**真正答案**$r_i$中出现的第$k$个**n-gram的数量**。当答案更短时，$P_n(C,R)$往往容易取得较高的分数。因此，通常设定一个惩罚因子BP：
$$
\mathrm{BP}=\left\{\begin{array}{l}
1, l_{c}>l_{r} \\
e^{1-\frac{l_{r}}{l_{c}}}, l_{c} \leq l_{r}
\end{array}\right.
$$
完整的BLEU计算方式如下：
$$
\mathrm{BLEU}=\mathrm{BP} \cdot \exp (\sum_{n=1}^{N} w_{n} \log P_{n})
$$
其中，$N$为gram的大小，例如若$N=4$，则称其为BLEU-4；$w_i$通常取$1/N$。



### 参考资料

[1] Liu S, Zhang X, Zhang S, et al. Neural machine reading comprehension: Methods and trends. Applied Sciences, 2019, 9(18): 3698.