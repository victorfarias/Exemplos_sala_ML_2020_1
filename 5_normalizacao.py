import numpy as np

def normalizacao01(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x-minx)/(maxx-minx)

def normalizacao_media(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    meanx = np.mean(x)
    return (x-meanx)/(maxx-minx)

def padronizar(x):
    meanx = np.mean(x)
    stdx = np.std(x)
    return (x-meanx)/stdx

def normalizar_X_media(X):
    for j in range(X.shape[1]):
        X[:,j] = normalizacao_media(X[:,j])

X = np.array(
    [[5 , 0.2],
     [2 , 0.1],
     [10, 0.7],
     [11, 0.5]]     
)

print()
norm = normalizacao01(X[:,1])
print(norm)
print(np.amax(norm))
print(np.amin(norm))

print()
norm1 = normalizacao_media(X[:,1])
print(norm1)
print(np.mean(norm1))

print()
norm2 = padronizar(X[:,1])
print(norm2)
print(np.mean(norm2))
print(np.std(norm2))


normalizar_X_media(X)
print(X)
