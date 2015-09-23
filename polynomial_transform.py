import numpy as np

def polynomial_transform(X):
    n = 3
    nSamples = X.shape[0]
    XX = X
    for j in range(n):
        for k in range(n):
            XX = np.hstack((np.reshape((X[:, 0]**j)*(X[:, 1]**k), (nSamples, 1)), XX))
    return XX