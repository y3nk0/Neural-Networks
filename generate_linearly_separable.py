__author__ = 'nicolas'
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(500, 2)

Y = (.5*X[:, 0] - .2*X[:, 1] > 0).astype(float)

X = X - .2*np.outer(Y, [-1, 1])

X -= [-1, 1]

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
plt.axis([-2, 4, -5, 3])
plt.axis('off')
plt.show()

np.savez('linearly_separable.npz', X=X, Y=Y)