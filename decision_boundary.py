import numpy as np
import matplotlib.pyplot as plt
from polynomial_transform import *

def decision_boundary(X, Y, w, b):
    x_min, x_max = -2, 4
    y_min, y_max = -5, 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                     np.arange(y_min, y_max, .05))

    if len(w) > 2:
        XX = polynomial_transform(np.vstack((xx.ravel(), yy.ravel())).T)
    else:
        XX = np.vstack((xx.ravel(), yy.ravel())).T
    Z = np.dot(XX, w) + b

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z > 0, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
    plt.axis([-2, 4, -5, 3])
    plt.draw()