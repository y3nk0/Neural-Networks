import numpy as np
import matplotlib.pyplot as plt
from decision_boundary import *
from baseFunctions import *
from polynomial_transform import *

dataset = 'double_moon'

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
    eps = 1./np.amax(np.sum(X**2, 1))
elif algorithm == 'polynomial_regression':
    X = polynomial_transform(X)
    eps = 20/np.amax(np.sum(X**2, 1))

nSamples = X.shape[0]
plt.ion()

w = np.zeros(X.shape[1])
b = 0.

T = np.zeros(nSamples)
C = np.zeros(nSamples)
sum_gradients = np.zeros(X.shape[1])
gradients = np.zeros(nSamples)

for i in range(100*nSamples):
    # Draw an example at random
    n = np.random.randint(nSamples)

    # Predicted outputs
    T[n] = sigm(np.dot(X[n], w) + b)
    C[n] = (T[n] > .5)

    sum_gradients -= gradients[n]*X[n]
    gradients[n] = Y[n] - T[n]
    sum_gradients += gradients[n]*X[n]

    w += eps*sum_gradients/nSamples
    b += eps*np.mean(gradients[n])

    if i%nSamples == 0:
        # Draw the decision boundary.
        plt.clf()
        plt.title('p = ' + str(X.shape[1]) + ', Iteration = ' + str(i/nSamples) + ', Error = ' + str(np.mean(Y != C)))
        decision_boundary(Xorig, Y, w, b)