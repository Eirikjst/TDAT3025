#
# Make an non-linear model that predicts head circumference given the persons age (in days)
#
# Observations: https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math as math

path = 'C:/Users/eirik/workspace python/TDAT3025/Read_csv_files/Read_CSV.py'
url = 'https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv'

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('Read_csv_files', path)
csv = loader.load_module('Read_csv_files')
data = csv.ReadCSV(url).read_csv_from_url()

#print(data.size)
x_data, y_data = np.mat(data[0:,[0]]), np.mat(data[0:,[1]])

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = (20*tf.sigmoid(-self.x)*(tf.matmul(self.x, self.W) + self.b))+31#tf.matmul(self.x, self.W) + self.b

        # Uses Mean Squared Error, although instead of mean, sum is used.
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.000000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(5000):
    session.run(minimize_operation, {model.x: x_data, model.y: y_data})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_data, model.y: y_data})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

compute_W, compute_b = W, b

session.close()

#Visulazation part
fig, ax = plt.subplots()

#ax.plot(x_data, y_data, label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('length')
ax.set_ylabel('weight')

class LinearRegressionModel_visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return (20*(1/(1+math.exp(x)))*((x*self.W) + self.b))+31

    # Uses Mean Squared Error, although instead of mean, sum is used.
    def loss(self, x, y):
        return np.sum(np.square(self.f(x) - y))


model = LinearRegressionModel_visualize(np.mat(compute_W), np.mat(compute_b))

#x = np.mat([[np.min(x_data)], [np.max(x_data)]])
#ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

print('loss (numpy):', model.loss(x_data, y_data))

ax.legend()
plt.show()
