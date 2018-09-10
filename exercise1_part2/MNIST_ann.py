import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()

x_train = x_train_.reshape(x_train_.shape[0], x_train_.shape[1]*x_train_.shape[2])
x_test = x_test_.reshape(x_test_.shape[0], x_test_.shape[1]*x_test_.shape[2])
y_train = np.zeros((y_train_.size, 10))
y_train[np.arange(y_train_.size), y_train_] = 1

batches = 200 # Divide training data into batches to speed up optimization
x_train_batches = np.split(x_train, batches)
y_train_batches = np.split(y_train, batches)

x_train, x_test  = x_train.astype('float32'), x_test.astype('float32')

y_test = np.zeros((y_test_.size, 10))
y_test[np.arange(y_test_.size), y_test_] = 1


class MNIST_ann_model:
    def __init__(self):
        
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])

        self.W1 = tf.Variable(tf.random_normal([784, 40]))
        self.b1 = tf.Variable(tf.zeros([40]))

        f1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.x, self.W1), self.b1))

        self.W2 = tf.Variable(tf.random_normal([40, 10]))
        self.b2 = tf.Variable(tf.zeros([10]))

        f2 = tf.nn.softmax(tf.nn.bias_add(tf.matmul(f1, self.W2), self.b2))

        # Loss
        self.loss = -tf.reduce_sum(self.y * tf.log(f2))

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f2,1), tf.argmax(self.y,1)), tf.float32))

model = MNIST_ann_model()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.AdamOptimizer(0.001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(250):
    for batch in range(batches):
        session.run(minimize_operation, {model.x: x_train_batches[batch], model.y: y_train_batches[batch]})

    accuracy, loss = session.run([model.accuracy, model.loss], {model.x: x_test, model.y: y_test})
    print('Epoch =',epoch,', accuracy = ', float(format((accuracy*100), '.2f')),'%, loss = ',loss)
    
    W1_, b1_, W2_, b2_ = session.run([model.W1, model.b1, model.W2, model.b2], {model.x: x_test, model.y: y_test})

session.close()



for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(W1_[:,i].reshape(28,28))
    plt.title(i)
    plt.xticks([])
    plt.yticks([])
    
plt.show()