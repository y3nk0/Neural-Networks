__author__ = 'nicolas'
import numpy as np
import matplotlib.pyplot as plt

# Generate 500 points between -pi/2 and pi/2
Theta1 = np.random.rand(500)*np.pi - np.pi/2

# Generate 500 points between pi/2 and 3pi/2
Theta2 = np.random.rand(500)*np.pi + np.pi/2

R = 2
C1 = [0, 0]
C2 = [.8, -1.8]

X1 = np.zeros((500, 2))
X1[:, 0] = R*np.cos(Theta1) + C1[0] + .6*np.random.rand(500)
X1[:, 1] = R*np.sin(Theta1) + C1[1] + .6*np.random.rand(500)
X2 = np.zeros((500, 2))
X2[:, 0] = R*np.cos(Theta2) + C2[0] + .6*np.random.rand(500)
X2[:, 1] = R*np.sin(Theta2) + C2[1] + .6*np.random.rand(500)

Y = np.zeros(1000)
Y[:500] = 0
Y[501:] = 1

X = np.concatenate((X1, X2))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
plt.axis([-2, 4, -5, 3])
plt.axis('off')
plt.show()

np.savez('double_moon.npz', X=X, Y=Y)