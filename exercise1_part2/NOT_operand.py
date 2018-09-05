import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = np.mat([[0.0], [1.0]])
y_data = np.mat([[1.0], [0.0]])

W_init = np.mat([[-5.2262573]])
b_init = np.mat([[2.3908339]])


def sigmoid(t):
    return 1 / (1 + np.exp(-t))

#Visulazation part
class SigmoidModel_visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return sigmoid(x * self.W + self.b)

    # Uses Cross Entropy
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


model = SigmoidModel_visualize(W_init, b_init)

fig, ax = plt.subplots()

ax.plot(x_data,
        y_data,
        'o',
        label='$(\\hat x^{(i)},\\hat y^{(i)})$'
        )

x = np.linspace(0, 1, 100).reshape(-1, 1)

ax.plot(x,
        model.f(x),
        label='$y = f(x) = \sigma(xW+b)$'
        )

ax.legend()
plt.show()