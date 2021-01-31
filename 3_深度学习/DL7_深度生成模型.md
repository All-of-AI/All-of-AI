## 深度学习7 深度生成模型

### 7.1 概率生成模型

概率生成模型，简称**生成模型(generative model)**，是概率统计和机器学习中的一类重要模型，指一系列用于**随机生成**可观测数据的模型。假设在一个连续的或离散的高维空间$\mathcal X$中，存在一个随机向量$\boldsymbol X$服从一个未知的数据分布$p_r(\boldsymbol x),\boldsymbol x \in \mathcal X$。生成模型是根据一些可观测的样本$\boldsymbol x^{(1)},\boldsymbol x^{(2)},\cdots,\boldsymbol x^{(N)}$来学习一个参数化的模型$p_\theta(\boldsymbol x)$来近似未知分布$p_r(\boldsymbol x)$，并可以用这个模型来生成一些样本，使得**生成样本和真实样本尽可能地相似**。

自然情况下， 直接建模$p_r(\boldsymbol x)$比较困难。**深度生成模型**就是利用**深度神经网络可以近似任意函数**的能力来建模一个复杂的分布$p_r(\boldsymbol x)$。假设一个随机向量$\boldsymbol Z$服从一个简单的分布$p(\boldsymbol z),z \in \mathcal Z$(例如标准正态分布)，我们使用一个深度神经网络$g: \mathcal Z \rightarrow \mathcal X$，并使得$g(\boldsymbol z)$服从$p_r(\boldsymbol x)$。

生成模型一般具有两个功能：**密度估计**和**样本生成**。

给定一组数据$\mathcal D=\{\boldsymbol x^{(i)}\}, 1 \leqslant i \leqslant N$，假设它们都是独立地匆匆相同的概率密度函数为$p_r(\boldsymbol x)$的未知分布中产生的。**概率密度估计(probabilistic density estimation)**是根据数据集$\mathcal D$来估计其概率密度函数$p_\theta(\boldsymbol x)$。在机器学习中，概率密度估计是一种非常典型的无监督学习问题。如果要建模的分布包含隐变量(如高斯混合模型)，就需要利用EM算法来进行密度估计。

生成样本就是给定义一个概率密度函数为$p_\theta(\boldsymbol x)$的分布，生成一些服从这个分布的样本，也称为采样。对于一个概率生成模型，在得到两个变量的局部条件概率$p_\theta(\boldsymbol z)$和$p_\theta(\boldsymbol x|\boldsymbol z)$之后，我们就可以生成数据$\boldsymbol x$。具体地，首先根据隐变量的先验分布$p_\theta(\boldsymbol z)$进行采样，得到样本$\boldsymbol z$，然后根据条件分布$p_\theta(\boldsymbol x|\boldsymbol z)$进行采样，得到$\boldsymbol x$。

因此，在生成模型中，**重点是估计条件分布**$p(\boldsymbol x|\boldsymbol z;\theta)$。

### 7.2 变分自编码器



### 7.3 生成式对抗网络



### 参考资料

[1] 邱锡鹏. 神经网络与深度学习. 2019.

