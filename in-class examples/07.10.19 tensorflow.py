# Disable GPU in tensotflow-gpu - https://stackoverflow.com/a/44552793/11511781
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import numpy
# import random

# Create Data from Multivariate Normal Distribution
mean1 = [-3, 0]
mean2 = [3, 0]

#       ROW 1   ROW 2
cov = [[2, 0], [0, 3]]

x1 = numpy.random.multivariate_normal(mean1, cov, 1000)
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)

X = numpy.concatenate((x1, x2))
X = numpy.concatenate((numpy.ones((2000, 1)), X), axis=1)

# First Half of the Data from Class 0, Second Half from Class 1 - Labels
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

data_holder = tf.placeholder(tf.float32, shape=[None, 2])
label_holder = tf.placeholder(tf.float32, shape=[None, 2])

hid_nodes = 2
out_nodes = 1

# Input layer
w0 = tf.Variable(tf.random_normal([2, hid_nodes]))
# Hidden Layer
w1 = tf.Variable(tf.random_normal([hid_nodes, hid_nodes]))
# Output layer
w2 = tf.Variable(tf.random_normal([hid_nodes, out_nodes]))

# Define Biases
b0 = tf.Variable(tf.random_normal([hid_nodes]))
b1 = tf.Variable(tf.random_normal([hid_nodes]))
b2 = tf.Variable(tf.random_normal([hid_nodes]))

# Create Layers
layer_1 = tf.add(tf.matmul(data_holder, w0), b0)
layer_1 = tf.math.sigmoid(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, w1), b1)
layer_2 = tf.math.sigmoid(layer_2)
out_layer = tf.matmul(layer_2, w2)+b2
outSigmoid = tf.math.sigmoid(out_layer)

# Computer Loss

# th.math.reduce_mean - computes the mean of elements across dimensions of a tensor
loss = tf.sqrt(tf.reduce_mean((outSigmoid - label_holder) ** 2))

alpha = 0.25
num_epochs = 2000

optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

# Running this Graph
init = tf.global_variables_initializer()

# Allocate GPU Memory - OMIT IF NOT RUNNING ON GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create tensorflow Session
sess = tf.Session(config=config)
sess.run(init)  # performs initialization

# EVERYTHING BEFORE THIS IS SETUP

# Loop Over Epochs
for epoch in range(num_epochs):
    sess.run(optimizer, feed_dict={data_holder: X, label_holder: Xc})

# Running The Network ib Forward (->) Direction ( NO TRAINING )
output = outSigmoid.eval(session=sess, feed_dict={data_holder: X})
treshOutput = (output > 0.5) * 1
error = sum(abs(treshOutput - Xc))

print("DONE")
