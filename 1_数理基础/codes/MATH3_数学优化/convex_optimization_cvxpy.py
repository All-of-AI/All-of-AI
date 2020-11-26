# 使用cvxpy库解决凸优化问题

import cvxpy as cvx

# 定义要被优化的变量
x = cvx.Variable()
y = cvx.Variable()

# 定义约束，可以包含等式约束和不等式约束
constraints = [x + y == 1, x - y >= 1]

# 定义优化目标
obj = cvx.Minimize((x - y) ** 2)

# 将优化目标和约束组成优化问题
prob = cvx.Problem(obj, constraints)

# 求解
prob.solve()
print("status: ", prob.status)  # optimal
print("optimal value: ", prob.value)  # 1.0
print("optimal variables: ", x.value, y.value)  # 1.0 1.570086213240983e-22
