## 自然语言处理5 文本摘要

### 5.1 文本摘要简介

随着互联网产生的文本数据越来越多，文本信息过载问题日益严重，对各类文本进行一个“降 维”处理显得非常必要，**文本摘要(text summarization)**便是其中一个重要的手段。文本摘要旨在**将文本或文本集合转换为包含关键信息的简短摘要**。文本摘要按照**输入类型**可分为**单文档摘要和多文档摘要**。单文档摘要从给定的一个文档中生成摘要，多文档摘要从给定的一组主题相关的文档中生成摘要。按照**输出类型**可分为**抽取式摘要和生成式摘要**。抽取式摘要从源文档中抽取关键句和关键词组成摘要，**摘要全部来源于原文**；生成式摘要根据原文，允许**生成新的词语、短语来组成摘要**。按照有无监督数据可以分为**有监督摘要和无监督摘要**。

常用的**文本摘要数据集**有DUC数据集、New York Times数据集、CNN / Daily Mail数据集、 Gigaword数据集、LCSTS数据集等。**文本摘要的结果通常使用ROUGE指标及其变体进行评价**。

### 5.2 TextRank抽提式文本摘要

TextRank[1]是一种**基于PageRank**的**无监督文本摘要**算法。在TextRank算法中，句子等价于网页，使用任意两个句子的相似性等价于网页转换概率，将相似性得分存储在一个矩阵中，类似于PageRank中的概率转移矩阵。

TextRank算法步骤如下：

(1) 把所有文章整合成**文本数据**；

(2) 接下来把文本分割成**单个句子**；

(3) 将为每个句子找到向量表示，通常是先使用word2vec预训练每个单词的词向量，然后**将一句话中的所有单词的word2vec进行平均**，从而得到一句话的向量表示；

(4) 计算句子向量间的**相似性(通常使用余弦相似度)**并存放在矩阵中；

(5) 将相似矩阵转换为以句子为节点、相似性得分为边的**图结构**，用于句子TextRank计算；

(6) 最后，**将一定数量的排名最高的句子构成最后的摘要**。

### 5.3 seq2seq与attention生成式文本摘要

基于seq2seq+attention结构的生成式文本摘要示意图如下：

![image-20200803110440798](images/image-20200803110440798.png)

原始文本(source text)中的单词$w_i$被逐步输入一个编码器(论文中使用的是单层双向LSTM)，产生一个编码器隐藏层序列$h_i$，其长度与原始文本长度相同。在解码时的第$t$个时间步中，解码器接受前一个单词的词向量(在训练时是参考摘要的词向量，在测试阶段是decoder上一步输出的词向量)，并产生解码器隐含状态$s_t$。**注意力分布**可使用如下公式计算得出：
$$
\begin{array}{l}
e_{i}^{t}=v^{T} \tanh (W_{h} h_{i}+W_{s} s_{t}+b_{\mathrm{attn}})\\
a^{t}=\operatorname{softmax}(e^{t})
\end{array}
$$
然后，注意力分布用于计算上下文向量$h_t^*$：
$$
h_t^*=\sum_i a_i^t h_i
$$
上下文向量$h_t^*$可以看作是**从源文本读取的内容的固定大小表示形式**，它与解码器状态$s_t$串联在一起，并通过两个线性层以产生词汇分布$P_{\text{vocab}}$：
$$
P_{\text{vocab}}=\operatorname{softmax}(V^{\prime}(V[s_{t}, h_{t}^{*}]+b)+b^{\prime})
$$
最终预测单词的概率分布就等于在词表上所计算得到的概率分布$P_{\text{vocab}}$，即$P(w)=P_{\text{vocab}}$。

在训练过程中，时间步$t$的损失为目标单词$w_t^*$的负对数似然，即$\text{loss}_t=-\log P(w_t^*)$，整个句子的损失为：
$$
\operatorname{loss}=\frac{1}{T} \sum_{t=0}^{T} \operatorname{loss}_{t}
$$
使用如上结构进行文本摘要时，存在**三个问题**：(1) **对事实细节的错误描述(例如2-0变为2-1)**；(2) **无法处理未知词汇(OOV)问题**；(3) **容易产生重复(repetition)**。

### 5.4 PGN生成式文本摘要

**PGN(pointer-generator network)**[2]在seq2seq+attention结构的基础上进行两个改进：(1) 引入**指针网络**，从原文中复制部分单词，从而**提高摘要准确性，并解决OOV问题**；(2) 利用**coverage机制**减少重复。

PGN的结构如下图所示：

![image-20200803113112581](images/image-20200803113112581.png)

在PGN中，注意力分布$a^t$和上下文向量$h_t^*$与原始seq2seq+attention结构的计算方式一致。此外，定义在时间步$t$的**生成概率**$p_{\text{gen}}$：
$$
p_{\mathrm{gen}}=\sigma(w_{h^{*}}^{T} h_{t}^{*}+w_{s}^{T} s_{t}+w_{x}^{T} x_{t}+b_{\mathrm{ptr}})
$$
其中为sigmoid函数。然后，$p_{\text{gen}}$被用作一个“软开关”，来选择**从词典中生成一个单词**，或者根据注意力分布$a^t$**从输入句子中复制一个单词**。对于每一个文档，其**拓展词表(extended vocabulary)**指原词表与当前文档中未在原词表中出现的词汇的并集。因此，生成单词$w$的概率分布是建立在拓展词表上的：
$$
P(w)=p_{\text{gen}} P_{\text{vocab}}(w)+(1-p_{\text{gen}})\sum_{i: w_{i}=w} a_{i}^{t}
$$
注意，当$w$是OOV词汇时，$P_{\text{vocab}}(w)$值为0；类似地，当$w$未出现在源文档中时，$\sum_{i: w_{i}=w} a_{i}^{t}$值也为0。

为了解决生成过程中出现的重复问题，引入coverage机制。在模型中，维持一个coverage向量$c^t$：
$$
c^{t}=\sum_{t^{\prime}=0}^{t-1} a^{t^{\prime}}
$$
直觉上，$c_i^t$表示从$0-t$时间步中，单词$w_i$**已经得到attention的程度**。其中，$c^0$是一个零向量，因为在第一个时间步中，源文档中的所有部分都没有被覆盖。

coverage向量被用于注意力机制的一个额外输入：
$$
e_{i}^{t}=v^{T}\tanh(W_{h} h_{i}+W_{s} s_{t}+w_{c} c_{i}^{t}+b_{\mathrm{attn}})
$$
上式可以确保**注意力机制的当前决定(选择下一个注意力分布)考虑到其先前的决定(隐含在$c_t$中)**。 这应该使注意力机制更容易**避免重复关注相同的位置，从而避免生成重复的文本**。

定义coverage损失来对**频繁attend相同的位置**这一现象进行惩罚：
$$
\text{covloss_t}=\sum_i \min(a_i^t,c_i^t)
$$
最终模型的损失函数为：
$$
\operatorname{loss}_{t}=-\log P(w_{t}^{*})+\lambda \sum_{i} \min(a_{i}^{t}, c_{i}^{t})
$$
其中$\lambda$为超参数。

### 5.5 BERT抽提式文本摘要

[3]

### 参考资料

[1] Mihalcea R, Tarau P. Textrank: Bringing order into text. Proceedings of the 2004 conference on empirical methods in natural language processing. 2004: 404-411.

[2] See A, Liu P J, Manning C D. Get to the point: Summarization with pointer-generator networks. arXiv preprint arXiv:1704.04368, 2017.

[3] Liu Y. Fine-tune BERT for extractive summarization. arXiv preprint arXiv:1903.10318, 2019.

[4] https://www.jiqizhixin.com/articles/2019-03-25-7

[5] https://www.cnblogs.com/motohq/p/11887420.html