import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_train = np.mat([[0,0],[0,1],[1,0],[1,1]])
y_train = np.mat([[0],[1],[1],[0]])

#Values from compute_XOR.py
W1_init = np.mat([[5.6879086, 7.4347506], [5.6881604, 7.436453]])
b1_init = np.mat([[-8.691843 , -3.4053555]])
W2_init = np.mat([[-13.824482], [13.048925]])
b2_init = np.mat([[-6.1208825]])

# Values taken from https://gitlab.com/ntnu-tdat3025/ann/visualize/blob/master/xor-operator.py
#W1_init = np.mat([[10.0, -10.0], [10.0, -10.0]])
#b1_init = np.mat([[-5.0, 15.0]])
#W2_init = np.mat([[10.0], [10.0]])
#b2_init = np.mat([[-15.0]])

# Visualize part
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

class SigmoidModel_visualize:
    def __init__(self, W1=W1_init.copy(), W2=W2_init.copy(), b1=b1_init.copy(), b2=b2_init.copy()):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    # First layer function
    def f1(self, x):
        return sigmoid(x * self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return sigmoid(h * self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


model = SigmoidModel_visualize()

fig = plt.figure("Logistic regression: the logical XOR operator")

plot1 = fig.add_subplot(121, projection='3d')

plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$h=$f1$(x)=\\sigma(x$W1$+$b1$)$")
plot1_h1 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))
plot1_h2 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))

plot1.plot(
    x_train[:, 0].A.squeeze(),
    x_train[:, 1].A.squeeze(),
    y_train[:, 0].A.squeeze(),
    'o',
    label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
    color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$h_1,h_2$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

plot2 = fig.add_subplot(222, projection='3d')

plot2_f2 = plot2.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=$f2$(h)=\\sigma(h $W2$+$b2$)$")

plot2_info = fig.text(0.8, 0.9, "")

plot2.set_xlabel("$h_1$")
plot2.set_ylabel("$h_2$")
plot2.set_zlabel("$y$")
plot2.legend(loc="upper left")
plot2.set_xticks([0, 1])
plot2.set_yticks([0, 1])
plot2.set_zticks([0, 1])
plot2.set_xlim(-0.25, 1.25)
plot2.set_ylim(-0.25, 1.25)
plot2.set_zlim(-0.25, 1.25)

plot3 = fig.add_subplot(224, projection='3d')

plot3_f = plot3.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$y=f(x)=$f2$($f1$(x))$")

plot3_info = fig.text(0.3, 0.03, "")

plot3.set_xlabel("$x_1$")
plot3.set_ylabel("$x_2$")
plot3.set_zlabel("$y$")
plot3.legend(loc="upper left")
plot3.set_xticks([0, 1])
plot3.set_yticks([0, 1])
plot3.set_zticks([0, 1])
plot3.set_xlim(-0.25, 1.25)
plot3.set_ylim(-0.25, 1.25)
plot3.set_zlim(-0.25, 1.25)

table = plt.table(
    cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], colWidths=[0.1] * 3, colLabels=["$x_1$", "$x_2$", "$f(x)$"], cellLoc="center", loc="lower right")

x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
h1_grid = np.empty([10, 10])
h2_grid = np.empty([10, 10])
f2_grid = np.empty([10, 10])
f_grid = np.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        h = model.f1([[x1_grid[i, j], x2_grid[i, j]]])
        h1_grid[i, j] = h[0, 0]
        h2_grid[i, j] = h[0, 1]
        f2_grid[i, j] = model.f2([[x1_grid[i, j], x2_grid[i, j]]])
        f_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])

plot1_h1 = plot1.plot_wireframe(x1_grid, x2_grid, h1_grid, color="lightgreen")
plot1_h2 = plot1.plot_wireframe(x1_grid, x2_grid, h2_grid, color="darkgreen")

plot1_info.set_text("W1$=\\left[\\stackrel{%.2f}{%.2f}\\/\\stackrel{%.2f}{%.2f}\\right]$\nb1$=[{%.2f}, {%.2f}]$" %
                    (model.W1[0, 0], model.W1[1, 0], model.W1[0, 1], model.W1[1, 1], model.b1[0, 0], model.b1[0, 1]))

plot2_f2 = plot2.plot_wireframe(x1_grid, x2_grid, f2_grid, color="green")

plot2_info.set_text("W2$=\\left[\\stackrel{%.2f}{%.2f}\\right]$\nb2$=[{%.2f}]$" % (model.W2[0, 0], model.W2[1, 0], model.b2[0, 0]))

plot3_f = plot3.plot_wireframe(x1_grid, x2_grid, f_grid, color="green")

plot3_info.set_text(
    "$loss = -\\frac{1}{n}\\sum_{i=1}^{n}\\left [ \\hat y^{(i)} \\log\\/f(\\hat x^{(i)}) + (1-\\hat y^{(i)}) \\log (1-f(\\hat x^{(i)})) \\right ] = %.2f$" %
    model.loss(x_train, y_train))

table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f([[0, 0]]))
table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f([[0, 1]]))
table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f([[1, 0]]))
table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f([[1, 1]]))

plt.show()