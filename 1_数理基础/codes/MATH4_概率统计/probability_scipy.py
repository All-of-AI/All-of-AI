# 使用scipy库实现概率统计中的基本概念及常用的分布

import math
from scipy import special
from scipy import stats

# 计算阶乘
print(math.factorial(20))

# 计算组合数
print(special.binom(5, 3))

# --------------------离散型随机变量--------------------
# 二项分布
X = stats.binom(10, 0.2)  # Declare X to be a binomial random variable X~Bin(10, 0.2)
print(X.pmf(3))  # P(X = 3)
print(X.cdf(4))  # P(X <= 4)
print(X.mean())  # E[X]
print(X.var())  # Var(X)
print(X.std())  # Std(X)
print(X.rvs())  # Get a random sample from X
print(X.rvs(10))  # Get 10 random samples form X

# 泊松分布
X = stats.poisson(2)  # Declare X to be a poisson random variable
print(X.pmf(3))  # P(X = 3)
print(X.rvs())  # Get a random sample from X

# 几何分布
X = stats.geom(0.75)  # Declare X to be a geometric random variable
print(X.pmf(3))  # P(X = 3)
print(X.rvs())  # Get a random sample from X

# --------------------连续型随机变量--------------------
# 正态分布
A = stats.norm(3, math.sqrt(16))  # Declare A to be a normal random variable
print(A.pdf(4))  # f(3), the probability density at 3
print(A.cdf(2))  # F(2), which is also P(A < 2)
print(A.rvs())  # Get a random sample from A

# 指数分布
B = stats.expon(4)  # Declare B to be a normal random variable
print(B.pdf(1))  # f(1), the probability density at 1
print(B.cdf(2))  # F(2) which is also P(B < 2)
print(B.rvs())  # Get a random sample from B

# beta分布
X = stats.beta(1, 3)  # Declare X to be a beta random variable
print(X.pdf(0.5))  # f(0.5), the probability density at 1
print(X.cdf(0.7))  # F(0.7) which is also P(X < 0.7)
print(X.rvs())  # Get a random sample from X
