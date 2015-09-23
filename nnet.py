import numpy as np
import matplotlib.pyplot as plt
from decision_boundary_nnet import *
from baseFunctions import *
from polynomial_transform import *

# Load the dataset
from baseFunctions import sigm
from decision_boundary_nnet import decision_boundary_nnet

dataset = 'double_moon'
#dataset = 'linearly_separable'

# Load the dataset
with np.load('./' + dataset + '.npz') as data:
    X = data['X']
    Y = data['Y']

h = 7

eps = .01

nSamples = X.shape[0]
C = np.zeros(nSamples)
plt.ion()

w_i = np.random.randn(X.shape[1], h)
w_o = np.zeros(h)
b_i = np.zeros(h)
b_o = 0.

for i in range(100*nSamples):
    # Draw an example at random
    n = np.random.randint(nSamples)

    # Predicted output
    input_hidden = np.dot(X[n], w_i) + b_i
    hidden = np.maximum(input_hidden, 0)
    output = sigm(np.dot(hidden, w_o) + b_o)
    C[n] = (output > .5)

    gradient_output = (Y[n] - output)
    grad_w_o = hidden*gradient_output
    grad_b_o = gradient_output
    grad_hidden = w_o*gradient_output
    grad_input_hidden = grad_hidden * (input_hidden > 0)
    grad_w_i = np.outer(X[n], grad_input_hidden)
    grad_b_i = grad_input_hidden

    w_o += eps*grad_w_o
    w_i += eps*grad_w_i
    b_o += eps*grad_b_o
    b_i += eps*grad_b_i

    if i%nSamples == 0:
        # Draw the decision boundary.
        plt.clf()
        plt.title('p = ' + str(h) + ', Iteration = ' + str(i/nSamples) + ', Error = ' + str(np.mean(Y != C)))
        decision_boundary_nnet(X, Y, w_o, w_i, b_o, b_i)
