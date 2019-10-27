import numpy
import pandas
import random
import matplotlib.pyplot

# Load Data from File
data  = pandas.read_excel("data1.xls", header= None)

X = pandas.DataFrame.to_numpy(data)

# Create Labels
Xc = numpy.zeros((1000,1))
Xc = numpy.concatenate((Xc, numpy.ones((1000,1))), axis=0)

# Plot the Data
matplotlib.pyplot.scatter(X[:1000,0], X[:1000,1], c='r', marker='.')
matplotlib.pyplot.scatter(X[1000:,0], X[1000:,1], c='b', marker='.')
matplotlib.pyplot.ion()
matplotlib.pyplot.show()

# Generate Initial Weights
random.seed()
w = numpy.transpose(numpy.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))

# Augmented Data Matrix
X = numpy.concatenate((numpy.ones((2000,1)), X), axis= 1)

# The Algorithm
for epoch in range(10):
    totalError = 0
    for i in range(2000):
        # Perceptron Algorithm
        # 1. Implement Calculation of the Output
        z = (numpy.sign(numpy.dot(X[i,:],w))+1)/2
        # 2. Calculate the Error
        error = Xc[i] - z
        # For Our Reporting Use
        totalError = totalError + abs(error)
        # 3. Update the Weights
        w = w + 0.1*error*X[i,:]

print("Slope is ", -w[1] / w[2], "\n")
print("Y Intercept is ", -w[0] / w[2])