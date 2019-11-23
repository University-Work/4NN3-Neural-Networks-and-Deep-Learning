import numpy as np
import math

data = np.loadtxt('magic04_space_delimited.txt',
                  delimiter=' ')  # Load DATA into matrix

# Pick a 1000 points for training
#    ROW      COLUMN
# [ from:to, from:to ]
X = data[11832:12832, 0:9]  # attributes
y = data[11832:12832, 10]  # labels

# Augmented Data Matrix
# Y-axis = 0, X-axis = 1
# * Add a column of 1 to the beggining of the matrix for BIAS
X = np.concatenate((np.ones((1000, 1)), X), axis=1)
# print(len(X))


def mlp(X, y):
    # Initialize thr weights
    w1 = np.transpose(np.zeros(len(X[0])))
    w2 = np.transpose(np.zeros(len(X[0])))
    w3 = np.transpose(np.zeros(len(X[0])))
    w4 = np.transpose(np.zeros(len(X[0])))
    w5 = np.transpose(np.zeros(len(X[0])))
    w6 = np.transpose(np.zeros(len(X[0])))

    alpha = 0.25  # Based on Optimization Theory
    outputError = np.empty((2000, 1))
    epoch = 100
    total_error = 0

    for epoch in range(epoch):
        for i in range((len(X))):
            # Forward Calculation
            z1 = 1 / (1 + math.exp(-np.dot(X[i, :], w1)))
            z2 = 1 / (1 + math.exp(-np.dot(X[i, :], w2)))
            z3 = 1 / (1 + math.exp(-np.dot(X[i, :], w3)))
            z4 = 1 / (1 + math.exp(-np.dot(X[i, :], w4)))
            z5 = 1 / (1 + math.exp(-np.dot(X[i, :], w5)))

            # Input vector for z6
            xhidden = np.array([1, z1, z2, z3, z4, z5, 1, 1, 1, 1])
            z6 = 1 / (1 + math.exp(-np.dot(xhidden, w6)))

            # Prediction
            prediction = round(z6, 0)  # One Option...
            outputError[i] = abs(y[i] - prediction)
            total_error = total_error + np.asscalar(outputError[i])

            # Backward Propagation
            # Update the Error
            error6 = z6 * (1 - z6) * (y[i] - z6)  # Err k - Output Node
            # Err j - Hidden Layer Node
            error1 = z1 * (1 - z1) * (error6 * w6[1])
            # Err j - Hidden Layer Node
            error2 = z2 * (1 - z2) * (error6 * w6[2])
            # Err j - Hidden Layer Node
            error3 = z2 * (1 - z3) * (error6 * w6[3])
            # Err j - Hidden Layer Node
            error4 = z2 * (1 - z4) * (error6 * w6[4])
            # Err j - Hidden Layer Node
            error5 = z2 * (1 - z5) * (error6 * w6[5])

            # Updating the Weights
            w6 = w6 + [alpha * error6, alpha * error6 * z1, alpha * error6 * z2,
                       alpha * error6 * z3, alpha * error6 * z4, alpha * error6 * z5, 1, 1, 1, 1]

            w1 = w1 + [alpha * error1, alpha * error1 * X[i, 1], alpha * error1 * X[i, 2],
                       alpha * error1 * X[i, 3], alpha * error1 * X[i, 4], alpha * error1 * X[i, 5], 
                       alpha * error1 * X[i, 6], alpha * error1 * X[i, 7], alpha * error1 * X[i, 8],
                       alpha * error1 * X[i, 9]]

            w2 = w2 + [alpha * error2, alpha * error2 * X[i, 1], alpha * error2 * X[i, 2],
                       alpha * error2 * X[i, 3], alpha * error2 * X[i, 4], alpha * error2 * X[i, 5], 
                       alpha * error2 * X[i, 6], alpha * error2 * X[i, 7], alpha * error2 * X[i, 8],
                       alpha * error2 * X[i, 9]]

            w3 = w3 + [alpha * error3, alpha * error3 * X[i, 1], alpha * error3 * X[i, 2],
                       alpha * error3 * X[i, 3], alpha * error3 * X[i, 4], alpha * error3 * X[i, 5], 
                       alpha * error3 * X[i, 6], alpha * error3 * X[i, 7], alpha * error3 * X[i, 8],
                       alpha * error3 * X[i, 9]]

            w4 = w4 + [alpha * error4, alpha * error4 * X[i, 1], alpha * error4 * X[i, 2],
                       alpha * error4 * X[i, 3], alpha * error4 * X[i, 4], alpha * error4 * X[i, 5], 
                       alpha * error4 * X[i, 6], alpha * error4 * X[i, 7], alpha * error4 * X[i, 8],
                       alpha * error4 * X[i, 9]]

            w5 = w5 + [alpha * error5, alpha * error5 * X[i, 1], alpha * error5 * X[i, 2],
                       alpha * error5 * X[i, 3], alpha * error5* X[i, 4], alpha * error5 * X[i, 5], 
                       alpha * error5 * X[i, 6], alpha * error5 * X[i, 7], alpha * error5 * X[i, 8],
                       alpha * error5 * X[i, 9]]

            # For Every Epoch
            print("Iteration... ", epoch + i, " Error = ", total_error, "      ", end="\r", flush=True)

    # print(xhidden.shape)
    # print(total_error)
    # print(w6.shape)
    # print(z6)

mlp(X, y)
