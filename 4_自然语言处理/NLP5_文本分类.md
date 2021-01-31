## 自然语言处理5 文本分类

**文本分类(text classification)**在自然语言处理领域有着广泛的应用，例如**垃圾邮件检测(spam detection)**、**情感分析(sentiment analysis)**、**语言识别(language id)**以及**新闻类别标注(news classification)**等。文本分类有着非常多的实现方法，每种实现方法中文本的表示也存在着很大的不同。但总的来说，文本分类是一个**监督学习问题**，需要数据样本以及其对应标签。

分类问题又可以分为**二分类**、**多分类**以及**多标签分类**等类型。其中的**多标签分类**指的是为每个数据标定数量不相同的标签，其由于类标签数量不确定、**类标签之间可能有相互依赖**、多标签的训练集比较难以获取等原因，难度是分类问题中最大的，因此也称为了分类问题的研究热点。目前主流的多标签分类的实现方式有如下几种：

(1) **不考虑标签之间的关联性**：将每个标签的出现与否视为二分类任务。

(2) **考虑标签的关联性**：分类器链、序列生成任务、通过**标签共现(组合)**的方式转换为多分类任务。

对于基于神经网络的分类器，实现多标签分类，只需**将输出层的softmax激活函数改为sigmoid激活函数**，对每个类别进行二分类即可。在这个过程中，每个标签之间不是独立的，其关联性也会被神经网络学习到。

### 5.1 基于朴素贝叶斯算法的文本分类

朴素贝叶斯算法是一种生成模型，其实现简单，常常用于**文本分类的baseline**。在朴素贝叶斯算法中，文本常用词袋模型进行表示，即每个文档表示为一个大小为$|V|$的向量，其中$|V|$为词汇表的大小，每一个维度表示对应下标的单词在文档中出现的次数：

<img src="images/image-20200417082234463.png" style="zoom:55%;" />

朴素贝叶斯分类器是一个概率模型，其理论基础是贝叶斯定理，“朴素”一词指的是**条件独立性假设**。对于文档$d$，朴素贝叶斯算法预测使得后验概率$P(c|d)$最大的类别$c\in C$。**朴素贝叶斯算法的过程如下所示**：

<img src="images/image-20200417082740628.png" style="zoom: 60%;" />

### 5.2 基于卷积神经网络的文本分类

与朴素贝叶斯算法等这类传统机器学习方法不同，使用卷积神经网络(CNN)进行文本分类需要将词使用**词向量(例如Word2Vec)**进行表示。CNN最初是针对图像设计的，其利用卷积核(convolving filters)来提取图像的局部特征。然而CNN也可以用于文本的特征提取，其通常被称为**TextCNN**。

Yoon Kim[2]在论文《Convolutional Neural Networks for Sentence Classification》中提出了如下所示的CNN网络结构用于文本分类任务：

![image-20200417084853983](images/image-20200417084853983.png)

整个网络仅使用了一个卷积层、一个池化层和一个全连接层。输入层为词向量，其可以一开始将词向量矩阵随机初始化，并在训练的过程中进行学习(作为实验的baseline)，也可以一直**保持静态**(即使用其他语料训练完成后便不再改变)，还可以预训练后在该网络训练的过程中进行**微调**(在实验中能够取得更好的效果)。该论文还将输入设置为两个通道，其均为词向量，但**其中一个通道保持静态，另一个通道可以在训练过程汇中进行微调**。

值得注意的一点是，由于每篇文档的长度不一样，应当在短文档后部**填充(padding)**全部由$0$组成的词向量，使得每篇文档等长的同时不影响卷积的结果。

该论文的实验使用的**数据集**如下：

<img src="images/image-20200417090043293.png" style="zoom:50%;" />

其中，$c$为目标类别个数，$l$为平均句子长度，$N$为数据集大小，即句子的个数，$|V|$为词汇表的大小，$|V_{pre}|$为预训练词向量中存在的词的个数，其比$|V|$要小，因为**可能去除了一些停用词或者低频词**。$Test$指测试集的划分大小，其中$CV$指的是该数据集没有标准的测试集划分，因此使用$10-fold$交叉验证。

超参数设定如下：使用ReLU激活函数，使用3, 4, 5大小的卷积核各100个，dropout率为0.5，使用系数为3的$L_2$正则化，mini-batch的大小设定为50。初始词向量使用Google News训练的word2vec，维度为300。

实验结果对比如下：

![image-20200417090816914](images/image-20200417090816914.png)

另一篇关于TextCNN的论文[3]提出了如下所示的网络结构，该结构与[2]基本一致：

<img src="images/image-20200417095418695.png" style="zoom:60%;" />

该论文进行了**大量的实验**来优化TextCNN中的参数，其中比较重要的一些**结论和建议**如下：

(1) 预训练word2vec或GloVe效果好于one-hot编码形式。

(2) 卷积核大小对实验结果有较大影响，一般取1~10，文本越长，可设置卷积核大小越大。

(3) 卷积核的数量对实验结果有较大的影响，一般取100~600 ，一般使用Dropout比率为0~0.5。

(4) 选用ReLU和tanh作为激活函数优于其他的激活函数。

(5) 1-max pooling在该实验中优于其他pooling策略。

(6) 正则化对实验结果有相对小的影响。

(7)当评估一个模型的性能时，考虑模型的方差是必要的。因此，评估模型时应当使用交叉验证，同时考虑模型每次结果的方差以及范围。

一些实验结果如下所示：

![image-20200417101103282](images/image-20200417101103282.png)

![image-20200417101154246](images/image-20200417101154246.png)

<img src="images/image-20200417101245937.png" style="zoom:45%;" />

<img src="images/image-20200417101315566.png" style="zoom:45%;" />

<img src="images/image-20200417101315566.png" style="zoom: 45%;" />

### 5.3 基于fastText的文本分类

Armand Joulin等人[4]提出了如下fastText模型，可用于快速的文本分类， 并且准确度接近state-of-art模型。

<img src="images/image-20200423180413505.png" style="zoom:50%;" />

其中$x_1,x_2,\cdots,x_N$是一个句子中的$N$个特征，可以是词向量，也可以是**N-gram特征**。这些词的特征表示(word representation)在隐含层(hidden)被平均为**句子的特征表示(sentence representation)**，然后被送入一个线性分类器。当输出空间很大时，论文使用**层次化softmax(hierarchical softmax)**对结果类别进行计算以减少计算复杂度。对于一个由$N$个文档祖晨的集合，该模型的目标是最小化如下所示的**损失函数(或负对数似然函数)**：
$$
-\frac{1}{N}\sum_{n=1}^{N}y_n\log(f(BAx_n))
$$
其中，$A$是一个look-up表，用于将输入特征$x_n$映射到其对应的嵌入向量；$B$是隐含层到输出层的参数矩阵，$f$是softmax激活函数。

### 5.4 基于循环神经网络的文本分类

使用循环神经网络(recurrent neural network, RNN)进行文本分类时，通常每个**时间步(time step)**的输入是文本中每个单词的词向量，将最后一个单词对应的RNN的输出通过全连接层+softmax的形式映射到类别的概率分布：

<img src="images/image-20200508142405388.png" style="zoom:40%;" />

还可以使用**双向结构**，能够在一定程度上提升分类效果。除了将最后时刻的状态作为**序列表示**之外，我们还可以**对整个序列的所有状态进行平均**，并用这个平均状态来作为整个序列的表示：

<img src="images/image-20200508142535364.png" style="zoom:40%;" />

本质上，RNN是在产生一个**句嵌入(sentence smbeeding)**，然后使用全连接+softmax的方式对该句嵌入进行分类。除了单向RNN和双向RNN以外，还可以**利用每个时间步的输出组合成一个新的向量**作为句嵌入。除了原始的RNN以外，还可以将每一层替换为LSTM或者GRU。

### 5.5 基于图卷积网络的文本分类



### 5.6 基于预训练模型的文本分类

使用预训练模型进行文本分类时，仅需要根据语言、规模等要求下载模型的预训练权重后完成下游任务的代码。以BERT[6]模型为例，完成自己的分类任务的步骤如下所示：

(1) **下载预训练模型**：

![image-20200507153418016](images/image-20200507153418016.png)

(2) **修改run_classifier.py代码**：继承DataProcessor并定义自己的数据处理类，在create_model函数中定义自己的下游任务。其他细节详见BERT文本分类代码。

### 参考资料

[1] Dan Jurafsky, H. Martin. Speech and Language Processing(3rd ed. draft).

[2] Kim Y. Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882, 2014.

[3] Zhang Y, Wallace B. A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820, 2015.

[4] Joulin A, Grave E, Bojanowski P, et al. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016.

[5] GCN文本分类

[6] Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.