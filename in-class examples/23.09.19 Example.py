#%% 
import numpy
import math
import random
import matplotlib.pyplot

# Create Data from Multivariate Normal Distribution
mean1 = [-3, 0]
mean2 = [3, 0]

#       ROW 1   ROW 2
cov = [[2, 0], [0, 3]]

x1 = numpy.random.multivariate_normal(mean1, cov, 1000)
matplotlib.pyplot.scatter(x1[:, 0], x1[:, 1], c='b', marker='.')
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)
matplotlib.pyplot.scatter(x2[:, 0], x2[:, 1], c='r', marker='.')
matplotlib.pyplot.ion()
matplotlib.pyplot.show()

X = numpy.concatenate((x1, x2))
X = numpy.concatenate((numpy.ones((2000, 1)), X), axis=1)

# First Half of the Data from Class 0, Second Half from Class 1
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

# Randomly Initialize the Weights
random.seed()
w1 = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))
w2 = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))
w3 = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))

for i in range(2000):
    # Forward Calculation
    z1 = 1/(1 + math.exp(-numpy.dot(X[i, :], w1)))
    z2 = 1/(1 + math.exp(-numpy.dot(X[i, :], w2)))
    # Input Vector for z3
    xhidden = [1, z1, z2]
    z3 = 1/(1 + math.exp(-numpy.dot(xhidden, w3)))

print('DONE')


#%%
