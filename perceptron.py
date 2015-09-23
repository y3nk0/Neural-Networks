import numpy as np
import matplotlib.pyplot as plt
from decision_boundary import *

# Load the dataset
with np.load('./data.npz') as data:
    X = data['X']
    Y = data['Y']

w = np.zeros(X.shape[1])
b = 0.
nSamples = X.shape[0]
plt.ion()

for i in range(100):
    # Predicted outputs
    T = np.sign(np.dot(X, w) + b)
    w += .01*np.dot(Y - T, X)/nSamples
    b += .01*np.mean(Y - T)

    # Draw the decision boundary.
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
    plt.axis([-2, 4, -5, 3])
    plt.title('Iteration = ' + str(i) + ', Error = ' + str(np.mean(Y != T)))
    plt.axis('off')
    decision_boundary(w, b)