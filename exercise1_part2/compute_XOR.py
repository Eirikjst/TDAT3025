import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

W1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="W1")
W2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="W2")

b1 = tf.Variable(tf.zeros([2]), name="b1")
b2 = tf.Variable(tf.zeros([1]), name="b2")

f1 = tf.sigmoid(tf.matmul(x_, W1) + b1)
f2 = tf.sigmoid(tf.matmul(f1, W2) + b2)

loss = tf.reduce_mean(( (y_ * tf.log(f2)) + ((1 - y_) * tf.log(1.0 - f2)) ) * -1)

minimize_operation = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(500000):
        sess.run(minimize_operation, feed_dict={x_: XOR_X, y_: XOR_Y})
        if i % 100000 == 0:
            print('Epoch ', i)
            print('f2:\n ', sess.run(f2, feed_dict={x_: XOR_X, y_: XOR_Y}))
            print('W1:\n ', sess.run(W1))
            print('b1:\n ', sess.run(b1))
            print('W2:\n ', sess.run(W2))
            print('b2:\n ', sess.run(b2))
            print('loss: ', sess.run(loss, feed_dict={x_: XOR_X, y_: XOR_Y}))

sess.close()