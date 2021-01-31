## 自然语言处理7 预训练模型

### 7.1 ELMo

预训练词向量(如word2vec和GloVe等)通常只能为一个单词产生一个特定的词向量，而忽略了该单词的**上下文(context)**关系，因而无法解决**一词多义**或**一义多词**的问题。ELMo(embeddings from language models)[1]本质上是一个深度双向LSTM模型，用于为一个句子中的每个单词生成上下文相关的词向量。将这些上下文相关词向量编码了单词的深层次语义和句法信息，因此当ELMo应用到许多NLP任务中，这些任务的效果相对于使用静态的词向量往往能得到很大的提升。

ELMo是整个输入句子的函数，其输出为句子中每个单词的上下文相关词向量。给定一个含有$N$个标记的序列$(t_1,t_2,\cdots,t_N)$，**前向语言模型(forward language model)**通过建模在给定之前的标记序列$(t_1,\cdots,t_{k-1})$下$t_k$的概率来计算该句子(标记序列)的概率：
$$
p\left(t_{1}, t_{2}, \cdots, t_{N}\right)=\prod_{k=1}^{N} p\left(t_{k} | t_{1}, t_{2}, \cdots, t_{k-1}\right)
$$

在ELMo之前的语言模型通常为第$k$个位置的单词(通过embedding等方式)计算出一个上下文无关的词表示$\mathbf x_k^{LM}$，然后将其送入一个$L$层的前向LSTM。在每个位置$k$，每一层LSTM会输出一个上下文相关的表示为$\overrightarrow{\mathbf h}_{k,j}^{LM}$，其中$j=1,2,\cdots,L$。最顶层的LSTM输出$\overrightarrow{\mathbf h}_{k,L}^{LM}$在下游任务被用来预测下一个标记(通过sottmax层等方式)，即$t_{k+1}$。

**反向语言模型(backward language model)**与前向语言模型的计算方向正好相反：
$$
p\left(t_{1}, t_{2}, \cdots, t_{N}\right)=\prod_{k=1}^{N} p\left(t_{k} | t_{k+1}, t_{k+2}, \cdots, t_{N}\right)
$$
因此在第$k$个位置，第$j$层LSTM的输出表示为$\overleftarrow{\mathbf h}_{k,j}^{LM}$。

**双向语言模型(biLM)**可以结合前向和反向的语言模型。其可以形式化表示为**最大化前向和反向语言模型的对数似然函数之和**：
$$
\sum_{k=1}^{N} (\log p(t_{k} | t_{1}, \cdots, t_{k-1} ; \Theta_{x}, \overrightarrow{\Theta}_{LSTM}, \Theta_{s})+\log p(t_{k} | t_{k+1}, \cdots, t_{N} ; \Theta_{x}, \overleftarrow{\Theta}_{L S T M}, \Theta_{s}))
$$
在ELMo中，将第$k$ 个位置的标记一个$L$层的双向语言模型，可以得到$2L+1$个词的表示：
$$
\begin{aligned}
R_{k} &=\{\mathbf{x}_{k}^{L M}, \overrightarrow{\mathbf{h}}_{k, j}^{L M}, \overleftarrow{\mathbf{h}}_{k, j}^{L M} | j=1, \cdots, L\} \\
&=\left\{\mathbf{h}_{k, j}^{L M} | j=0, \cdots, L\right\}
\end{aligned}
$$
其中，${\mathbf h}_{k,0}^{LM}$代表$\mathbf{x}_{k}^{L M}$，${\mathbf h}_{k,0}^{LM}=[\overrightarrow{\mathbf h}_{k,j}^{LM};\overleftarrow{\mathbf h}_{k,j}^{LM}]$。

对于下游任务，ELMo将上述词的表示的向量集合使用一个单独的词向量$\mathbf{ELMo}_k=E(R_k;\Theta_e)$进行表示。最简单的情况是直接选出**最上层的词表示**作为最终结果，即$E(R_k)=\mathbf h^{LM}_{k,L}$，TagLM以及CoVe便是使用了这样的方法。更一般的方法是使用参数学习的方式来表示最终的上下文相关词向量：
$$
\mathbf{ELMo}_{k}^{t a s k}=E\left(R_{k} ; \Theta^{t a s k}\right)=\gamma^{t a s k} \sum_{j=0}^{L} s_{j}^{t a s k} \mathbf{h}_{k, j}^{L M}
$$
其中，$s^{task}$用于调节每一层的词表示$\mathbf h^{LM}_{k,j}$的权重，$\gamma^{task}$用于调节整个ELMo向量的权重。考虑到每一层的激活值会有不同的分布，可以在每一层后面添加一个层归一化(layer normalization)。

给定一个目标NLP任务的预训练biLM和一个有监督的体系结构，使用biLM改进任务模型是一个简单的过程。我们只需运行biLM并记录每个单词在所有层中的表示，然后将每个层产生的词表示合并为最终单一的词表示向量$\mathbf{ELMo}_k^{task}$。在这之后，便可以直接将该单一的词表示向量送入下游任务中，如将一句话中的单词逐个输入RNN中进行文本分类、问答等任务。

作为一种双向语言模型，ELMo**预训练的方式**是根据一个单词左右两边的单词来预测当前单词。

ELMo在多个NLP任务及数据集上的实验结果如下所示：

![image-20200420174149047](images/image-20200420174149047.png)

### 7.2 Transformer

在transformer之前的序列模型均采用了**循环层(recurrent layer)和卷积层(convolution layer)**，其中效果最好的模型均采用了**注意力(attention)**机制。transformer摒弃了全部循环层和卷积层，只基于注意力机制，实现了**计算的并行化**。原论文[2]采用机器翻译以及其他NLP任务来验证transformer的有效性。

transformer的模型结构如下所示，其中左半部分是多个编码器，右半部分是多个解码器：

<img src="images/image-20200420153815269.png" style="zoom:50%;" />

##### Encoder and Decoder Stacks

**编码器(encoder)**是由$N=6$个编码器层组成的栈式结构，其中每个编码器层中有两个sub-layers，即**多头注意力机制(multi-head attentin)**层和posotion-wise**全连接前馈神经网络**层。

每一个sub-layer都采用了**残差连接(residual connection)**以及**层归一化(layer normalization)**。因此，每个sub-layer的输出为$\text{LayerNorm}(x+\text{Sublayer}(x))$。为了方便进行残差连接，模型中所有sub-layers的输出维度，以及embedding层的输出维度，均为$d_{model}=512$。

**解码器(decoder)**也是由$N=6$个解码器层组成的栈式结构，但解码器层比编码器层多一个sub-layer，即解码器输出时用到的多头注意力机制，该注意力机制与seq2seq中的注意力机制类似。解码器中与编码器对应位置的多头注意力机制使用了遮盖，保证了在位置$i$的预测仅仅依赖于在$i$之前已知的预测。

##### Attention

注意力函数可以描述为**将查询(query)和一组键(key)-值(value)对映射到输出**，其中查询、键、值和输出都是向量。**输出被计算为值的加权和**，其中分配给每个值的权重由查询的兼容函数和相应的键计算。

论文中的attention机制称为“scaled dot-product attention”，如下图所示：

<img src="images/image-20200420184046759.png" style="zoom:50%;" />

其输入包括queries(维度为$d_k$)、keys(维度为$d_k$)以及values(维度为$d_v$)。当并行处理多个输入的时候，可以将其组成三个矩阵$Q,K,V$，并且以下式来计算注意力输出：
$$
\text{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$
两种最常用的注意力函数分别是加性注意力(additive attention)和点积注意力(dot-product attention)。论文中的注意力使用了点积注意力，并且除以$\sqrt{d_k}$。与加性注意力相比，点积注意力计算更快，并且在实际应用中更加节省空间。值得注意的是，$QK^T$除以$\sqrt{d_k}$的原因如下：当$d_k$的值比较大时，点积的增长速度很快，导致sotfmax函数来到梯度非常小的区域。因此，因子$\sqrt{d_k}$的作用是**控制sotfmax函数的值的范围，减少梯度消失的问题**。

论文提出，将输入线性映射到$h$个不同的空间内并且使用多个并行的注意力机制能够提高模型性能。该方式称为多头注意力机制，如下图所示：

<img src="images/image-20200420201802904.png" style="zoom:50%;" />

多头注意力机制可以形式化地表示为：
$$
\begin{aligned}
\text {MultiHead}(Q, K, V) &=\text {Concat}\left(\text {head}_{1}, \ldots, \text {head}_{\mathrm{h}}\right) W^{O} \\
\text {where head}_{\mathrm{i}} &=\text {Attention}(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V})
\end{aligned}
$$
该式可以利用下图进行理解：

![image-20200420203933069](images/image-20200420203933069.png)

假设模块的输入为$X$(即原始句子的词嵌入序列，维度为$(len,d_{model})$)或者$R$(即上一个编码器模块传来的输入，维度也为$(len,d_{model})$)。图中的$X$或$R$等价于原文公式中的$Q,K,V$，首先分别乘以$3\times h$个不同的权重矩阵(维度分别为$(d_{model},d_k),(d_{model},d_k),(d_{model},d_v)$)以映射到$3\times h$个不同的空间，然后分别按照每一个注意力头的$Q,K,V$进行单头注意力计算，产生每个注意力头的结果$Z_i$，维度为$(len,d_v)$。将$h$个$Z_i$连接在一起，得到维度为$(len,hd_v)$大小的矩阵。将连接后的矩阵$Z$与参数矩阵$W^O$(维度为$(hd_v,d_{model})$)相乘，得到最终的结果为$Z$，维度为$(len,d_{model})$。原文取$h=8,d_k=d_v=d_{model}/h=64$。

transformer中的多头注意力机制有如下三种不同的方式：

(1) 在encoder-decoder attention中，queries来自于**上一个解码器层的输出**，keys和values来自于**每一个编码器层的输出**。这种方式使得解码器输出的每一步都能关注编码器输入的每一个位置。

(2) 在编码器中，**所有的keys、values和queries都来自于同一个矩阵**，即最初的词向量序列输入$X$或者上一个编码器层的输出$R$。

(3) 在解码器中，每一步解码器能够关注当前步以及当前步之前的向量。我们需要保留解码器中输出句子每一步左边的信息流来维持其**自回归(auto-regression)**的特性。该方法通过**遮盖(mask)**实现。

##### Position-wise Feed-Forward Networks

在多头注意力层后，加入一层全连接前馈神经网络层，该网络层由两次线性映射以及一个ReLU非线性映射组成，即$FFN(x)=max(0,xW_1+b_1)W_2+b_2$。在该网络中，输入和输出的维度均为$d_{model}=512$，中间层的维度为$d_{ff}=2048$。

##### Embeddings and Softmax

与其他序列模型相似，transformer编码器的inputs和解码器的outputs(都是输入)均是学习得到的词向量，维度均为$d_{model}$。sotfmax层在解码器之后，结合一个线性变换层用于预测当前单词的概率分布。

##### Positional Encoding

由于模型中不含循环层以及卷积层，为了使模型得以利用句子中的顺序信息，必须向输入中加入相对或绝对的位置信息。文章通过给编码器和解码器的输入词向量$X$加入**位置编码(positional encodings)**来实现顺序信息的引入。位置编码的维度与输入词向量相同，均为$d_{model}$。最终输入编码器和解码器的向量为**原始词向量与其对应位置编码之和**。

在论文中，位置编码采用如下形式：
$$
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin (pos / 10000^{2 i / d_{\text{model}}}) \\
P E_{(\text {pos}, 2 i+1)} &=\cos (pos / 10000^{2 i / d_{\text{model}}})
\end{aligned}
$$
其中，$pos$是每个单词在句子中的位置(position)，$i$代表每个词向量的第$i$个维度(dimension)。

下表分析了self-attention与循环层和卷积层相比时间复杂度的差异：

![image-20200420222402891](images/image-20200420222402891.png)

实验结果如下：

![image-20200420222424199](images/image-20200420222424199.png)

### 7.3 BERT

**BERT(bidirectional encoder representations from transformers)**是一种语言模型，用于从无标签文本中学习词的**深度双向表示**，这个过程称为**预训练(pre-train)**。在BERT预训练后，可以附加下游任务，通过**微调(fine-tuning)**的方式来完成NLP任务，如句子级别的自然语言推断、解释任务以及单词级别的NER、QA等任务。

目前有两种将预训练语言表示应用于下游任务的策略，即**基于特征(feature-based)**的方法及**微调(fine-tuning)**。二者的典型代表分别是**ELMo**和**GPT**。两种方法在预训练过程中使用的目标函数都是相同的，都是使用单向的语言模型取学习一般的语言表示(ELMo是由两个单向的LSTM拼接实现的，本质还是单向模型)。

BERT的论文[3]认为，ELMo和GPT这类模型的**单向性**限制了下游任务的效果。BERT通过两种方式进行预训练：

(1) **masked language model, MLM**：将输入中的标记(token)随机mask一部分，使其不可见，然后让模型去基于这些被遮盖的标记的上下文来预测这些单词的id。

(2) **next sentence prediction, NSP**：给出两个句子A和B，让模型去判断B是否是A的下一个句子。

BERT模型的两种预训练方式以及其在下游任务中的应用示意图如下所示：

![image-20200425201254058](images/image-20200425201254058.png)

BERT模型与ELMo和GPT的区别如下所示：

![image-20200426101648120](images/image-20200426101648120.png)

#### 7.3.1 模型结构

BERT模型本质上是一个多层的双向transformer的编码器。设transformer编码器层的数量为$L$，隐含层单元的数量为$H$，注意力头的数量为$A$。论文中提出了两种尺寸的BERT，即$\bold{BERT}_{\bold{BASE}}(L=12,H=768,A=12)$，以及$\bold{BERT}_{\bold{LARGE}}(L=12,H=1024,A=16)$。

为了让BERT能处理更多类型的下游任务，输入数据可以包含两种形式，即一个单独的句子以及一个句子对(例如question和answer)。文章使用wordpiece embedding对输入单词进行表示，词汇表大小为30,000。每个句子的开头标记都是一个特殊类别标记“[CLS]”，**最终对应这个标记的隐含状态便是用于分类任务的整个句子的表示**。包含两个输入句子的桔子堆被合并为一个序列，并使用两种方法对这两个句子进行区分：(1) 在两个句子之间添加一个特殊标记[SEP]；(2) 为每个标记添加一个学习得到的嵌入，该嵌入表明每个标记属于句子A还是句子B。记$E$为输入标记的嵌入，$C \in \mathbb R^H$为[CLS]对应的最终隐含层向量，第$i$个输入标记对应的最终隐含层向量为$T_i \in \mathbb R^H$。

对于一个给定的标记，**其输入表示为token embedding、segment embedding和position embedding之和**。

![image-20200425210720499](images/image-20200425210720499.png)

#### 7.3.2 预训练BERT

BERT通过masked language model和next sentence prediction两种方式进行预训练。

(1) **masked language model, MLM**：为了训练一个深度双向的表示，论文简单地将输入标记序列中的一些单词随机遮盖后让BERT完成预测这些单词的任务。被遮盖的单词最终的隐含层向量被送入一个输出的softmax层，通过训练的方式不断提高模型预测的准确率。实验遮盖了**每个序列中15%的wordpiece标记**。

MLM产生的一个缺点是，[MASK]标记导致了预训练与微调的不匹配，因为[MASK]标记不出现在微调过程中。为了解决这个问题，对于被随机选中第$i$个masked标记，其**以80%的可能性被替换为[MASK]标记，10%的可能性被替换为一个随机的标记，10%的可能性保持不变**。然后，$T_i$被用于预测原始标记。

(2) **next sentence prediction, NSP**：许多例如QA、NLI的下游任务基于理解句子之间的关系，这种关系无法直接被语言模型捕获。NSP任务可用于捕获句子之间的关系，其具体做法如下：从每个预训练样本中选择句子A和B时，有50%的可能性句子B确实是跟在句子A后面的句子(被标记为**IsNext**)，还有50%的可能性句子B是从语料库中随机选取的(被标记为**NotNext**)。在BERT中，$C$([CLS]标记对应位置的最终隐含层向量)用于NSP。这个预训练方法对QA和NLI任务非常有效。

预训练的数据来自BooksCorpus(300M words)和Wikipedia(2,500M words)。使用像维基百科这样文档级别的语料库非常有帮助，因为其能够使得BERT更好地捕获长程依赖。

#### 7.3.3 微调BERT

仅需要改变输入和输出，BERT便能用于完成很多种类的NLP任务。在微调的过程中，BERT中的全部参数都会以端到端的方式进行微调。下图展示了对于不同任务，BERT输入和输出的变化：

![image-20200426100129779](images/image-20200426100129779.png)

![image-20200426100154337](images/image-20200426100154337.png)

#### 7.3.4 实验结果

在不同NLP任务上的实验结果如下：

(1) GLUE

![image-20200426100615444](images/image-20200426100615444.png)

(2) SQuAD 1.1

<img src="images/image-20200426100706463.png" style="zoom:50%;" />

(3) SQuAD 2.0

<img src="images/image-20200426100809998.png" style="zoom:50%;" />

(4) SWAG

<img src="images/image-20200426100834569.png" style="zoom:50%;" />

(5) 预训练方式调整

<img src="images/image-20200426101035301.png" style="zoom:50%;" />

(6) 网络结构调整

<img src="images/image-20200426101208031.png" style="zoom:50%;" />

(7) 命名实体识别

<img src="images/image-20200426101335649.png" style="zoom:50%;" />

### 7.4 ERNIE

#### 7.4.1 ERNIE 1.0

Google提出的BERT模型，利用Transformer的多层self-attention双向建模能力，在各项NLP下游任务中都取得了很好的成绩。但是，BERT模型主要是**聚焦在针对字或者英文word粒度的完形填空学习上面**，没有充分利用**训练数据当中词法结构，语法结构，以及语义信息**去学习建模。比如“我要买苹果手机”，BERT模型将“我 要 买 苹 果 手 机”每个字都统一对待，**在预训练时随机进行遮盖(mask)**，丢失了“苹果手机”是一个很火的名词这一信息，这个是**词法信息**的缺失。同时“我 + 买 + 名词”是一个非常明显的购物意图的句式，BERT没有对此类**语法结构**进行专门的建模，如果预训练的语料中只有“我要买苹果手机”，“我要买华为手机”，哪一天出现了一个新的手机牌子比如栗子手机，而这个手机牌子在预训练的语料当中并不存在，**没有基于词法结构以及句法结构的建模，对于这种新出来的词是很难给出一个很好的向量表示的**，而ERNIE 1.0[4]通过进行**实体(entity)和短语(phrase)的masking**，极大地增强了通用语义表示能力，在多项任务中均取得了大幅度超越BERT的效果。

**ERNIE和BERT不同的masking策略**如下所示：

![image-20200527092631883](images/image-20200527092631883.png)

与BERT相同，ERNIE的整体网络架构也使用Transformer的编码器。Transformer可以通过**自注意力机制**捕捉句子中每个标记的上下文信息，并**生成一系列上下文嵌入(contextual embedding)**。

ERNIE使用先验知识来增强预训练语言模型，其提出了一种**多阶段的知识masking策略**，将短语和实体层次的知识整合到语言表达中，而**不是直接加入知识嵌入**。下图描述了句子的不同masking级别：

![image-20200527103621711](images/image-20200527103621711.png)

(1) **Basic-level masking**：Basifc-level masking是第一个阶段，它把一个句子看作一个基本语言单位的序列，对于英语，基本语言单位是**单词**，对于汉语，基本语言单位是**汉字**。在训练过程中，随机屏蔽15%的基本语言单元，并使用句子中的其他基本单元作为输入，训练一个Transformer的编码器来预测屏蔽单元。基于Basic-level masking，我们可以得到一个基本的单词表示。**因为它是在基本语义单元的随机掩码上训练的，所以很难对高层语义知识进行完全建模**。这个过程与BERT相同。

(2) **Phrase-level masking**：Phrase-level masking是第二个阶段。**短语**是作为概念单位的一小组单词或字符。**对于英语，我们使用词汇分析和分块工具来获取句子中短语的边界**，并**使用一些依赖于语言的切分工具来获取其他语言(如汉语)中的单词/短语信息**。在Phrase-level masking阶段，**仍然使用基本语言单元作为训练输入**，与随机基本单元mask不同，这次我们随机选择句子中的几个短语，对同一短语中的**所有基本单元进行mask和预测**。**在此阶段，短语信息被编码到单词嵌入中**。

(3) **Entity-level masking**：Entity-level masking是第三个阶段。**名称实体**包含个人、地点、组织、产品等，可以用适当的名称表示。它可以是抽象的，也可以是物理存在的。通常**实体在句子中包含重要信息**。和Phrase-level masking阶段一样，**首先分析句子中的命名实体**，然后屏蔽和预测实体中的所有语言单元。经过三个阶段的学习，得到了一个**由丰富的语义信息增强的词表示**。

ERNIE使用中文维基百科、百度百科、百度新闻和百度贴吧的综合语料库进行预训练。

此外，ERNIE**使用多轮对话修改BERT中的NSP(next sentence prediction)任务**：

![image-20200527105452130](images/image-20200527105452130.png)

在ERNIE中，NSP任务变为了**DLM(dialogue language model)**任务。使用dialogue embedding来区分不同的对话角色，可以表示多轮对话。与BERT中的**MLM(masked language model)**一样，masks被应用于强制模型来预测查询和响应条件下的丢失单词。此外，通过用随机选择的句子替换查询Q或响应R来生成假样本。该模型用于判断多回合对话是真是假。DLM任务帮助ERNIE学习对话中的隐含关系，这也增强了模型学习语义表示的能力。

以上便是ERNIE 1.0相对于BERT所做的改进。

#### 7.4.2 ERNIE 2.0

5]是一个多任务持续学习的预训练框架，其构建了三种类型的无监督任务，取得了较ERNIE 1.0更好的效果。

### 7.5 Transformer-XL与XLNet

[6]

### 参考资料

[1] Peters M E, Neumann M, Iyyer M, et al. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Advances in neural information processing systems. 2017: 5998-6008.

[3] Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[4] Sun Y, Wang S, Li Y, et al. Ernie: Enhanced representation through knowledge integration. arXiv preprint arXiv:1904.09223, 2019.

[5] Sun Y, Wang S, Li Y, et al. Ernie 2.0: A continual pre-training framework for language understanding. arXiv preprint arXiv:1907.12412, 2019.

[6] Transformer-XL与XLNet

7] https://jalammar.github.io/illustrated-transformer/

[8] https://blog.csdn.net/PaddlePaddle/article/details/102713947