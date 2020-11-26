# python实现LU分解

import numpy as np
from scipy import linalg
import numpy as np


def lu_decomposition(A):
    n = len(A[0])
    L = np.zeros([n, n])
    U = np.zeros([n, n])
    for i in range(n):
        L[i][i] = 1
        if i == 0:
            U[0][0] = A[0][0]
            for j in range(1, n):
                U[0][j] = A[0][j]
                L[j][0] = A[j][0] / U[0][0]
        else:
            for j in range(i, n):  # U
                temp = 0
                for k in range(0, i):
                    temp = temp + L[i][k] * U[k][j]
                U[i][j] = A[i][j] - temp
            for j in range(i + 1, n):  # L
                temp = 0
                for k in range(0, i):
                    temp = temp + L[j][k] * U[k][i]
                L[j][i] = (A[j][i] - temp) / U[i][i]
    return L, U


A = np.array([[4., -1., -1., 0., 0., 0., 0., 0.],
              [-1., 4., -1., -1., 0., 0., 0., 0.],
              [-1., -1., 4., -1., -1., 0., 0., 0.],
              [0., -1., -1., 4., -1., -1., 0., 0.],
              [0., 0., -1., -1., 4., -1., -1., 0.],
              [0., 0., 0., -1., -1., 4., -1., -1.],
              [0., 0., 0., 0., -1., -1., 4., -1.],
              [0., 0., 0., 0., 0., -1., -1., 4.]])
L, U = lu_decomposition(A)
print(L.tolist())
print(U.tolist())
