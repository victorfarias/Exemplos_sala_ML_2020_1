import numpy as np

# Declarando vetores e matrizes
v = np.array([1, 2, 3])
M = np.array([[1,2], [3,4]])

# Funções interessantes
sum = np.sum(M)
print(sum)
mean = np.mean(M)
transpose_M = M.T
inversa_M = np.linalg.inv(M)

# Operações com vetores
v1 = np.array([1,2,3])
v2 = np.array([3,5,7])

print(v1+v2)
residuo = v1 - v2
RSS = residuo*residuo
print(RSS)

# Adicionar uma coluna de 1's na matriz M
print(M.shape)
ones = np.ones(M.shape[0])
print(ones)

X_ = np.c_[ones, M]
print(X_)

# Carregar conjunto de dados
data = np.loadtxt("./data/advertising.csv", skiprows=1, delimiter=",")
X = data[:,1:4]
y = data[:, 4]
print(X)
print(y)