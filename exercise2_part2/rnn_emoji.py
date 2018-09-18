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
        #self.out_state = tf.reshape(self.out_state, [-1, 128])

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.einsum('bts,se->bte', lstm, W), b)  # b: batch, t: time, s: state, e: encoding

        #logits = tf.nn.bias_add(tf.matmul(tf.reshape(lstm, [-1, 13]), W), b)

        # Predictor
        self.f = tf.nn.softmax(logits)
        #self.f = tf.reshape(self.f, [-1, 13])

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)



char_encodings = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ' ' - 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'a' - 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'c' - 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'f' - 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 'h' - 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 'l' - 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 'm' - 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 'n' - 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 'o' - 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 'p' - 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 'r' - 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 's' - 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 't' - 12
]

'''
char_encodings = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # ' ' - 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'a' - 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'c' - 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'f' - 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'h' - 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'l' - 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'm' - 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'n' - 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'o' - 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'p' - 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 'r' - 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 's' - 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 't' - 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 'hat ' - 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 'rat ' - 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 'cat ' - 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 'flat' - 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 'matt' - 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 'cap ' - 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 'son ' - 18
]
'''

index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't']#, 'hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son ']

x_train = [
    char_encodings[0],  # ' '
    char_encodings[4], char_encodings[1], char_encodings[12], char_encodings[0], # 'hat '
    char_encodings[0],  # ' '
    char_encodings[10], char_encodings[1], char_encodings[12], char_encodings[0], # 'rat '
    char_encodings[0],  # ' '
    char_encodings[2], char_encodings[1], char_encodings[12], char_encodings[0], # 'cat '
    char_encodings[0],  # ' '
    char_encodings[3], char_encodings[5], char_encodings[1], char_encodings[12], # 'flat'
    char_encodings[0],  # ' '
    char_encodings[6], char_encodings[1], char_encodings[12], char_encodings[12], # 'matt'
    char_encodings[0],  # ' '
    char_encodings[2], char_encodings[1], char_encodings[9], char_encodings[0], # 'cap '
    char_encodings[0],  # ' '
    char_encodings[11], char_encodings[8], char_encodings[7], char_encodings[0], # 'son '
    ] # 'hat  rat  cat  flat matt cap  son '


y_train = [
    char_encodings[4], char_encodings[1], char_encodings[12], char_encodings[0], # 'hat '
    char_encodings[0],  # ' '
    char_encodings[10], char_encodings[1], char_encodings[12], char_encodings[0], # 'rat '
    char_encodings[0],  # ' '
    char_encodings[2], char_encodings[1], char_encodings[12], char_encodings[0], # 'cat '
    char_encodings[0],  # ' '
    char_encodings[3], char_encodings[5], char_encodings[1], char_encodings[12], # 'flat'
    char_encodings[0],  # ' '
    char_encodings[6], char_encodings[1], char_encodings[12], char_encodings[12], # 'matt'
    char_encodings[0],  # ' '
    char_encodings[2], char_encodings[1], char_encodings[9], char_encodings[0], # 'cap '
    char_encodings[0],  # ' '
    char_encodings[11], char_encodings[8], char_encodings[7], char_encodings[0], # 'son '
    char_encodings[0],  # ' '
]
'''
y_train = [
    char_encodings[13], char_encodings[13], char_encodings[13], char_encodings[13], # 'hat '
    char_encodings[0],  # ' '
    char_encodings[14], char_encodings[14], char_encodings[14], char_encodings[14], # 'rat '
    char_encodings[0],  # ' '
    char_encodings[15], char_encodings[15], char_encodings[15], char_encodings[15], # 'cat '
    char_encodings[0],  # ' '
    char_encodings[16], char_encodings[16], char_encodings[16], char_encodings[16], # 'flat'
    char_encodings[0],  # ' '
    char_encodings[17], char_encodings[17], char_encodings[17], char_encodings[17], # 'matt'
    char_encodings[0],  # ' '
    char_encodings[18], char_encodings[18], char_encodings[18], char_encodings[18], # 'cap '
    char_encodings[0],  # ' '
    char_encodings[18], char_encodings[18], char_encodings[18], char_encodings[18], # 'son '
    char_encodings[0],  # ' '
    ] # 'hat rat cat flat matt cap son'
'''

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
    session.run(minimize_operation, {model.batch_size: 4, model.x: [x_train], model.y: [y_train], model.in_state: zero_state})

    if epoch % 100 == 9:
        print("epoch", epoch)
        print("loss", session.run(model.loss, {model.batch_size: 1, model.x: [x_train], model.y: [y_train], model.in_state: zero_state}))

        # Generate characters from the initial characters 'rt '
        state = session.run(model.in_state, {model.batch_size: 1})
        text = 'rt '
        y, state = session.run([model.f, model.out_state], {model.batch_size: 4, model.x: [[char_encodings[10]]], model.in_state: state}) # 's'
        y, state = session.run([model.f, model.out_state], {model.batch_size: 4, model.x: [[char_encodings[12]]], model.in_state: state}) # 't'
        y, state = session.run([model.f, model.out_state], {model.batch_size: 4, model.x: [[char_encodings[0]]], model.in_state: state}) # ' '
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[1]]], model.in_state: state}) # 'a'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[2]]], model.in_state: state}) # 'c'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[3]]], model.in_state: state}) # 'f'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[4]]], model.in_state: state}) # 'h'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[5]]], model.in_state: state}) # 'l'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[6]]], model.in_state: state}) # 'm'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[7]]], model.in_state: state}) # 'n'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[8]]], model.in_state: state}) # 'o'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[9]]], model.in_state: state}) # 'p'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 7, model.x: [[char_encodings[10]]], model.in_state: state}) # 'r'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: [[char_encodings[11]]], model.in_state: state}) # 's'
        #y, state = session.run([model.f, model.out_state], {model.batch_size: 7, model.x: [[char_encodings[12]]], model.in_state: state}) # 't'
        text += index_to_char[y.argmax()]
        for c in range(50):
            y, state = session.run([model.f, model.out_state], {model.batch_size: 4, model.x: [[char_encodings[y.argmax()]]], model.in_state: state})
            text += index_to_char[y[0].argmax()]
            #print(state[0][0].argmax())
        print(text)

session.close()