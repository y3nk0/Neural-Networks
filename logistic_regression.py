import numpy as np
import matplotlib.pyplot as plt
from decision_boundary import *
from baseFunctions import *
from polynomial_transform import *

# Load the dataset
dataset = 'double_moon'
#dataset = 'linearly_separable'

# Load the dataset
with np.load('./' + dataset + '.npz') as data:
    X = data['X']
    Y = data['Y']


algorithms = ['perceptron', 'logistic_regression', 'polynomial_regression']
algorithm = algorithms[1]

Xorig = X
if algorithm == 'perceptron':
    eps = 1.
elif algorithm == 'logistic_regression':
    eps = 5.
elif algorithm == 'polynomial_regression':
    X = polynomial_transform(X)
    eps = 100

nSamples = X.shape[0]
plt.ion()

w = np.zeros(X.shape[1])
b = 0.

for i in range(100):
    if algorithm == 'perceptron':
        # Predicted outputs
        T = (np.dot(X, w) + b) > 0
        C = T
    else:
        # Predicted outputs
        T = sigm(np.dot(X, w) + b)
        C = (T > .5)

    w += eps*np.dot(Y - T, X)/nSamples
    b += eps*np.mean(Y - T)

    # Draw the decision boundary.
    plt.clf()
    plt.title('p = ' + str(X.shape[1]) + ', Iteration = ' + str(i) + ', Error = ' + str(np.mean(Y != C)))
    decision_boundary(Xorig, Y, w, b)