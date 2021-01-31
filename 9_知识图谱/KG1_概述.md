## 知识图谱1 概述

### 1.1 知识图谱简介

知识图谱是**事实的结构化表征**，包括了实体(entities)、关系(relationships)及语义描述(semantic descriptions)。**实体**可能是客观对象和抽象概念，**关系**表示这些实体之间的关系，**对实体和关系的语义描述**包含预定义的**类型(types)和属性(properties)**。

Knowledge graph(KG)和knowledge base(KB)几乎可以看做同义词，只不过knowledge base是知识库，而knowledge graph则是**基于知识库的图结构**。基于resource description framework(RDF)可以**用三元组的形式表示知识**，即**(head, relation, tail)或者(subject, predicate, object)**。知识图谱是一个有向图，其中**结点表示实体，边表示关系**。

<img src="images/image-20201123203817862.png" style="zoom:40%;" />

知识图谱最近的研究方向集中在**知识表示学习(KRL)以及知识图谱嵌入(KGE)**，主要**将实体和关系映射到低维向量中同时捕获它们的语义**。知识图谱的各类任务如下所示：

<img src="images/image-20201123203041357.png" style="zoom:45%;" />

#### 1.1.1 知识表示学习(KRL)

KRL通常可以分为4个不同的方面：**表征空间(representation space)，打分函数(scoring function)，编码模型(encoding models)以及辅助信息(auxiliary information)**。关于一个KRL模型的具体组件包括：

- 用来表征关系和实体的表征空间，包括point-wise空间、流形、复数的向量空间、高斯分布以及离散空间；
- 用来衡量事实三元组的合理性，一般有基于距离的和基于相似度的打分函数；
- 用来学习表征和关系交互的编码模型，主要包括线性/双线性模型、分解机以及神经网络；
- 融入到embedding方法的辅助信息，主要考虑文本、视觉和类型信息。

#### 1.1.2 知识获取(KA)

KA任务主要分为3个子任务，即**KGC(知识图谱补全)、实体发现以及关系抽取**。第一个任务用来完善已有的KG，其他两个则是从文本中发掘新知识，即新的实体和关系。

KGC任务包括**基于embedding的排序、关系路径推理、基于规则的推理以及元关系学习**；实体发现主要包括识别、排歧、类型确定以及对齐；关系抽取主要使用attention机制、GCNs、对抗学习、强化学习、残差学习以及迁移学习。

#### 1.1.3 时序知识图谱

时序KG通过融合时间信息进行表征学习，通常包括四个研究方向：时序embedding、动态实体、时序关系依赖以及时序逻辑推理。

#### 1.1.4 知识感知应用

知识感知应用包括NLU，QA，推荐系统以及其他日常任务，通过注入知识改善表示学习。

### 参考资料

[1] Ji S, Pan S, Cambria E, et al. A survey on knowledge graphs: Representation, acquisition and applications. arXiv preprint arXiv:2002.00388, 2020.

[2] https://zhuanlan.zhihu.com/p/38056557