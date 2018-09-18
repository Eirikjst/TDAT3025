import numpy as np
import tensorflow as tf

class LongShortTermMemoryModel:
    def __init__(self, encodings_size):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [])  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encodings_size])  # Shape: [batch_size, max_time, encodings_size]
        self.y = tf.placeholder(tf.float32, [None, None, encodings_size])  # Shape: [batch_size, max_time, encodings_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, encodings_size]))
        b = tf.Variable(tf.random_normal([encodings_size]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.einsum('bts,se->bte', lstm, W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


char_encodings = [
    [1, 0, 0, 0, 0, 0, 0, 0], # ' ' - 0
    [0, 1, 0, 0, 0, 0, 0, 0], # 'h' - 1
    [0, 0, 1, 0, 0, 0, 0, 0], # 'e' - 2
    [0, 0, 0, 1, 0, 0, 0, 0], # 'l' - 3
    [0, 0, 0, 0, 1, 0, 0, 0], # 'o' - 4
    [0, 0, 0, 0, 0, 1, 0, 0], # 'w' - 5
    [0, 0, 0, 0, 0, 0, 1, 0], # 'r' - 6
    [0, 0, 0, 0, 0 ,0 ,0, 1], # 'd' - 7
]
index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = [
    char_encodings[0], # ' '
    char_encodings[1], # 'h'
    char_encodings[2], # 'e'
    char_encodings[3], # 'l'
    char_encodings[3], # 'l'
    char_encodings[4], # 'o'
    char_encodings[0], # ' '
    char_encodings[5], # 'w'
    char_encodings[4], # 'o'
    char_encodings[6], # 'r'
    char_encodings[3], # 'l'
    char_encodings[7], # 'd'
    ] # ' hello world'
  
y_train = [
    char_encodings[1], # 'h'
    char_encodings[2], # 'e'
    char_encodings[3], # 'l'
    char_encodings[3], # 'l'
    char_encodings[4], # 'o'
    char_encodings[0], # ' '
    char_encodings[5], # 'w'
    char_encodings[4], # 'o'
    char_encodings[6], # 'r'
    char_encodings[3], # 'l'
    char_encodings[7], # 'd'
    char_encodings[0], # ' '
    ] # 'hello world '

model = LongShortTermMemoryModel(np.shape(char_encodings)[1])

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

# Initialize model.in_state
zero_state = session.run(model.in_state, {model.batch_size: 1})

for epoch in range(500):
    session.run(minimize_operation, {model.batch_size: 1, model.x: [x_train], model.y: [y_train], model.in_state: zero_state})

    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", session.run(model.loss, {model.batch_size: 1, model.x: [x_train], model.y: [y_train], model.in_state: zero_state}))

        # Generate characters from the initial characters ' h'
        state = session.run(model.in_state, {model.batch_size: 1})
        text = ' h'
        y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[0]]], model.in_state: state})  # ' '
        y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[1]]], model.in_state: state})  # 'h'
        text += index_to_char[y.argmax()]
        for c in range(50):
            y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[y.argmax()]]], model.in_state: state})
            text += index_to_char[y[0].argmax()]
        print(text)

session.close()

# output
'''
epoch 9
loss 1.5990232
 hllo lo wo l wo  lwo  wo  l wo  lwo  wo  l wo  lwo
epoch 19
loss 1.7436216
 hlloorlorlorlloworlorlloworlloworlloworlloworlloworl
epoch 29
loss 1.2698756
 helloworllorlddd wrllorlddd wlorlddldd wlorlddldd wl
epoch 39
loss 0.74633884
 hellowrdd wrdd helld wordd d helld wordd d helld wor
epoch 49
loss 0.23553514
 hello worlld worlld worlld worlld worlld worlld worl
epoch 59
loss 0.06285876
 hello world  hello world  hello world  hello world
epoch 69
loss 0.02949515
 hello world  hello world  hello world  hello world
epoch 79
loss 0.0142739555
 hello world  hello world  hello world  hello world
epoch 89
loss 0.69045436
 hello wollo wolld wollo wolld wollo wolld wollo woll
epoch 99
loss 0.36134952
 hellloworld  d llo world  world  world  world  world
epoch 109
loss 0.005271667
 hello world   d hello world  d ld  hello world  d he
epoch 119
loss 0.0025187526
 hello world  d hello world  d hello world  d hello w
epoch 129
loss 0.0013444112
 hello world  d hello world  d hello world  d hello w
epoch 139
loss 0.0007515588
 hello world  d hello world  d hello world  d hello w
epoch 149
loss 0.0004295595
 hello world  d hello world  d hello world  d hello w
epoch 159
loss 0.00024802447
 hello world  d hello world  d hello world  d hello w
epoch 169
loss 0.00014371744
 hello world  d hello world  d hello world  d hello w
epoch 179
loss 8.334978e-05
 hello world  d hell  world  hello world  hello world
epoch 189
loss 4.8250113e-05
 hello world  d hell  world  hello world  hello world
epoch 199
loss 2.7957523e-05
 hello world  hello world  hello world  hello world
epoch 209
loss 1.626624e-05
 hello world  hello world  hello world  hello world
epoch 219
loss 9.660371e-06
 hello world  hello world  hello world  hello world
epoch 229
loss 5.8855876e-06
 hello world  hello world  hello world  hello world
epoch 239
loss 3.7498553e-06
 hello world  hello world  hello world  hello world
epoch 249
loss 2.5677523e-06
 hello world  hello world  hello world  hello world
epoch 259
loss 1.8525304e-06
 hello world  hello world  hello world  hello world
epoch 269
loss 1.4055166e-06
 hello world  hello world  hello world  hello world
epoch 279
loss 1.1075074e-06
 hello world  hello world  hello world  hello world
epoch 289
loss 9.28702e-07
 hello world  hello world  hello world  hello world
epoch 299
loss 7.796973e-07
 hello world  hello world  hello world  hello world
epoch 309
loss 7.0022827e-07
 hello world  hello world  hello world  hello world
epoch 319
loss 6.306928e-07
 hello world  hello world  hello world  hello world
epoch 329
loss 5.4129004e-07
 hello world  hello world  hello world  hello world
epoch 339
loss 4.816882e-07
 hello world  hello world  hello world  hello world
epoch 349
loss 4.4195366e-07
 hello world  hello world  hello world  hello world
epoch 359
loss 4.3202e-07
 hello world  hello world  hello world  hello world
epoch 369
loss 3.9228547e-07
 hello world  hello world  hello world  hello world
epoch 379
loss 3.6248454e-07
 hello world  hello world  hello world  hello world
epoch 389
loss 3.326836e-07
 hello world  hello world  hello world  hello world
epoch 399
loss 3.2274997e-07
 hello world  hello world  hello world  hello world
epoch 409
loss 3.028827e-07
 hello world  hello world  hello world  hello world
epoch 419
loss 2.8301542e-07
 hello world  hello world  hello world  hello world
epoch 429
loss 2.8301542e-07
 hello world  hello world  hello world  hello world
epoch 439
loss 2.8301542e-07
 hello world  hello world  hello world  hello world
epoch 449
loss 2.8301542e-07
 hello world  hello world  hello world  hello world
epoch 459
loss 2.6314814e-07
 hello world  hello world  hello world  hello world
epoch 469
loss 2.2714646e-07
 hello world  hello world  hello world  hello world
epoch 479
loss 2.2714646e-07
 hello world  hello world  hello world  hello world
epoch 489
loss 2.2714646e-07
 hello world  hello world  hello world  hello world
epoch 499
loss 2.0107842e-07
 hello world  hello world  hello world  hello world
'''