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
# matplotlib.pyplot.ion()
matplotlib.pyplot.show()

X = numpy.concatenate((x1, x2))
X = numpy.concatenate((numpy.ones((2000, 1)), X), axis=1)

# First Half of the Data from Class 0, Second Half from Class 1 - Labels
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

# Randomly Initialize the Weights
random.seed()
w1 = numpy.transpose(numpy.array([random.random() - 0.5, random.random() - 0.5, random.random() - 0.5]))
w2 = numpy.transpose(numpy.array([random.random() - 0.5, random.random() - 0.5, random.random() - 0.5]))
w3 = numpy.transpose(numpy.array([random.random() - 0.5, random.random() - 0.5, random.random() - 0.5]))

alpha = 0.25  # Based on Optimization Theory
outputError = numpy.empty((2000, 1))
for epoch in range(2000):
    totalError = 0
    for i in range(2000):
        # Forward Calculation
        z1 = 1 / (1 + math.exp(-numpy.dot(X[i, :], w1)))
        z2 = 1 / (1 + math.exp(-numpy.dot(X[i, :], w2)))
        # Input Vector for z3
        xhidden = [1, z1, z2] # 1 is added for bias
        z3 = 1 / (1 + math.exp(-numpy.dot(xhidden, w3)))

        # Prediction
        prediction = round(z3, 0)  # One Option...
        outputError[i] = abs(Xc[i] - prediction)
        totalError = totalError + numpy.asscalar(outputError[i])

        # Backward Propagation
        # Update the Error
        error3 = z3 * (1 - z3) * (Xc[i] - z3)  # Err k - Output Node
        error1 = z1 * (1 - z1) * (error3 * w3[1])  # Err j - Hidden Layer Node
        error2 = z2 * (1 - z2) * (error3 * w3[2])  # Err j - Hidden Layer Node

        # Updating the Weights
        w3 = w3 + [alpha * error3, alpha * error3 * z1, alpha * error3 * z2]
        w1 = w1 + [alpha * error1, alpha * error1 * X[i, 1], alpha * error1 * X[i, 2]]
        w2 = w1 + [alpha * error2, alpha * error2 * X[i, 1], alpha * error2 * X[i, 2]]

        # For Every Epoch
        print("Iteration... ", epoch + i, " Error = ", totalError, "      ", end="\r", flush=True)

print('DONE')
