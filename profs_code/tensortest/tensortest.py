import tensorflow as tf
import numpy

mean1 = [-3,0]
mean2 = [3,0]
cov = [[2,0],[0,3]]
x1 = numpy.random.multivariate_normal(mean1, cov, 10000)
x2 = numpy.random.multivariate_normal(mean2, cov, 10000)
X = numpy.concatenate((x1,x2))

xtest = numpy.random.multivariate_normal([6,0], cov, 500)
xctest = numpy.zeros(500)


Xc = numpy.zeros((10000,1))
Xc = numpy.concatenate((Xc,numpy.ones((10000,1))))

data_holder = tf.placeholder(tf.float32, shape = [None,2])
label_holder = tf.placeholder(tf.float32, shape = [None,1])

hid_nodes = 2 
out_nodes = 1 

# Define weights 
w0 = tf.Variable(tf.random_normal([2, hid_nodes])) 
w1 = tf.Variable(tf.random_normal([hid_nodes, hid_nodes])) 
w2 = tf.Variable(tf.random_normal([hid_nodes, out_nodes])) 

# Define biases 
b0 = tf.Variable(tf.random_normal([hid_nodes])) 
b1 = tf.Variable(tf.random_normal([hid_nodes])) 
b2 = tf.Variable(tf.random_normal([out_nodes])) 

# Create layers 
layer_1 = tf.add(tf.matmul(data_holder, w0), b0) 
layer_1 = tf.math.sigmoid(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, w1), b1) 
layer_2 = tf.math.sigmoid(layer_2) 
out_layer = tf.matmul(layer_2, w2) + b2
outSigmoid = tf.math.sigmoid(out_layer)


# Compute loss 
#loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=out_layer, labels=label_holder))
#loss = tf.nn.sigmoid(tf.sqrt(tf.reduce_mean((out_layer - label_holder)**2)))
loss = tf.sqrt(tf.reduce_mean((outSigmoid - label_holder)**2))

#loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = out_layer, labels = label_holder)
# Create optimizer 
learning_rate = 0.25 
num_epochs = 20000
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize variables 
init = tf.global_variables_initializer()

# allocate gpu memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#create a Tensorflow session
sess = tf.Session(config = config)


sess.run(init) 
# Loop over epochs 
for epoch in range(num_epochs):
    sess.run(optimizer, feed_dict={data_holder: X, label_holder: Xc})


# get the output of the graph after training
output = outSigmoid.eval(session = sess, feed_dict={data_holder: X})
threshOutput = (output > 0.5)*1
error = sum(abs(threshOutput - Xc))

print("Done")