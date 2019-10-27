# Have to Use tensorflow 1.14 b/c version 2.0 has some issues
import tensorflow as tf

b = tf.Variable(tf.zeros((100,)))
W = tf.Variable(tf.random.uniform((784, 100), -1, 1))
x = tf.compat.v1.placeholder(tf.float32, (1, 784))

# h = ReLU(Wx+b)
h = tf.nn.relu(tf.matmul(x, W) + b)
print(h)