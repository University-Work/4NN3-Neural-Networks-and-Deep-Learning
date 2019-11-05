import numpy as np
import math
import random
import matplotlib.pyplot as plt

data = np.loadtxt(open('Data_for_UCI_named.csv'), delimiter=',')

X = data[0:7500, 0:12]
X = np.concatenate((X, np.ones((7500, 1))), axis=1)  # add bias column
y = data[0:7500, 13]

test_X = data[7501:10000, 0:12]
test_X = np.concatenate((test_X, np.ones((2499, 1))),
                        axis=1)  # add bias column
test_y = data[7501:10000, 13]


def mlp_3(X, y):
    random.seed()
    w1 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w2 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w3 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w4 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    alpha = 0.25
    epoch = 1000
    outputError = np.empty((7500, 1))

    for epoch in range(epoch):
        totalError = 0
        for i in range(len(X)):
            # forward calculation
            z1 = 1/(1 + math.exp(-np.dot(X[i, :], w1)))
            z2 = 1/(1 + math.exp(-np.dot(X[i, :], w2)))
            z3 = 1/(1 + math.exp(-np.dot(X[i, :], w3)))

            xhidden = [1, z1, z2, z3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            z4 = 1/(1 + math.exp(-np.dot(xhidden, w4)))

            # prediction
            prediction = round(z4, 0)
            outputError[i] = abs(y[i] - prediction)
            totalError = totalError + np.asscalar(outputError[i])

            # backward propagation
            # update the error
            error4 = z4*(1-z4)*(y[i] - z4)  # output node
            error1 = z1*(1-z1)*(error4*w4[1])
            error2 = z2*(1-z2)*(error4*w4[2])
            error3 = z3*(1-z3)*(error4*w4[3])

            w4 = w4 + [alpha*error4, alpha*error4 *
                       z1, alpha*error4*z2, alpha*error4*z3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            w1 = w1 + [alpha*error1, alpha*error1*X[i, 1], alpha*error1*X[i, 2],
                       alpha*error1*X[i, 3], alpha*error1 *
                       X[i, 4], alpha*error1*X[i, 5],
                       alpha*error1*X[i, 6], alpha*error1 *
                       X[i, 7], alpha*error1*X[i, 8],
                       alpha*error1*X[i, 9], alpha*error1*X[i, 10], alpha*error1*X[i, 11], alpha*error1*X[i, 12]]

            w2 = w2 + [alpha*error2, alpha*error2*X[i, 1], alpha*error1*X[i, 2],
                       alpha*error2*X[i, 3], alpha*error2 *
                       X[i, 4], alpha*error2*X[i, 5],
                       alpha*error2*X[i, 6], alpha*error2 *
                       X[i, 7], alpha*error2*X[i, 8],
                       alpha*error2*X[i, 9], alpha*error1*X[i, 10], alpha*error1*X[i, 11], alpha*error1*X[i, 12]]

            w3 = w3 + [alpha*error3, alpha*error3*X[i, 1], alpha*error3*X[i, 2],
                       alpha*error3*X[i, 3], alpha*error3 *
                       X[i, 4], alpha*error1*X[i, 5],
                       alpha*error2*X[i, 6], alpha*error3 *
                       X[i, 7], alpha*error3*X[i, 8],
                       alpha*error3*X[i, 9], alpha*error1*X[i, 10], alpha*error1*X[i, 11], alpha*error1*X[i, 12]]

            # print("Error: ", totalError)
            # print("Weight Vector: ", [w1, w2, w3, w4])
            return np.array([w1, w2, w3, w4, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def mlp_4(X, y):
    random.seed()
    w1 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w2 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w3 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w4 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    w5 = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))

    alpha = 0.25
    epoch = 1000
    outputError = np.empty((7500, 1))

    for epoch in range(epoch):
        totalError = 0
        for i in range(len(X)):
            # forward calculation
            z1 = 1/(1 + math.exp(-np.dot(X[i, :], w1)))
            z2 = 1/(1 + math.exp(-np.dot(X[i, :], w2)))
            z3 = 1/(1 + math.exp(-np.dot(X[i, :], w3)))
            z4 = 1/(1 + math.exp(-np.dot(X[i, :], w4)))

            xhidden = [1, z1, z2, z3, z4, 1, 1, 1, 1, 1, 1, 1, 1]
            z5 = 1/(1 + math.exp(-np.dot(xhidden, w5)))

            # prediction
            prediction = round(z4, 0)
            outputError[i] = abs(y[i] - prediction)
            totalError = totalError + np.asscalar(outputError[i])

            # backward propagation
            # update the error
            error5 = z5*(1-z5)*(y[i] - z5)  # output node
            error1 = z1*(1-z1)*(error5*w5[1])
            error2 = z2*(1-z2)*(error5*w5[2])
            error3 = z3*(1-z3)*(error5*w5[3])
            error4 = z4*(1-z4)*(error5*w5[3])

            w5 = w5 + [alpha*error5, alpha*error5 *
                       z1, alpha*error5*z2, alpha*error5*z3, alpha*error5*z4, 0, 0, 0, 0, 0, 0, 0, 0]
            w1 = w1 + [alpha*error1, alpha*error1*X[i, 1], alpha*error1*X[i, 2],
                       alpha*error1*X[i, 3], alpha*error1 *
                       X[i, 4], alpha*error1*X[i, 5],
                       alpha*error1*X[i, 6], alpha*error1 *
                       X[i, 7], alpha*error1*X[i, 8],
                       alpha*error1*X[i, 9], alpha*error1*X[i, 10], alpha*error1*X[i, 11], alpha*error1*X[i, 12]]

            w2 = w2 + [alpha*error2, alpha*error2*X[i, 1], alpha*error1*X[i, 2],
                       alpha*error2*X[i, 3], alpha*error2 *
                       X[i, 4], alpha*error2*X[i, 5],
                       alpha*error2*X[i, 6], alpha*error2 *
                       X[i, 7], alpha*error2*X[i, 8],
                       alpha*error2*X[i, 9], alpha*error1*X[i, 10], alpha*error1*X[i, 11], alpha*error1*X[i, 12]]

            w3 = w3 + [alpha*error3, alpha*error3*X[i, 1], alpha*error3*X[i, 2],
                       alpha*error3*X[i, 3], alpha*error3 *
                       X[i, 4], alpha*error1*X[i, 5],
                       alpha*error2*X[i, 6], alpha*error3 *
                       X[i, 7], alpha*error3*X[i, 8],
                       alpha*error3*X[i, 9], alpha*error1*X[i, 10], alpha*error1*X[i, 11], alpha*error1*X[i, 12]]

            w4 = w4 + [alpha*error4, alpha*error4*X[i, 1], alpha*error4*X[i, 2],
                       alpha*error4*X[i, 3], alpha*error4 *
                       X[i, 4], alpha*error4*X[i, 5],
                       alpha*error4*X[i, 6], alpha*error4 *
                       X[i, 7], alpha*error4*X[i, 8],
                       alpha*error4*X[i, 9], alpha*error4*X[i, 10], alpha*error4*X[i, 11], alpha*error4*X[i, 12]]

            # print("Error: ", totalError)
            # print("Weight Vector: ", [w1, w2, w3, w4])
            return np.array([w1, w2, w3, w4, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_mpl(test_X, test_y):
    errors1 = []
    errors2 = []
    w1 = mlp_3(X, y)
    w2 = mlp_4(X, y)

    for i in range(len(test_X)):
        prediction1 = np.dot(test_X[i, :], w1)
        prediction2 = np.dot(test_X[i, :], w2)
        error1 = test_y[i] - prediction1
        error2 = test_y[i] - prediction2

        errors1.append(error1)
        errors2.append(error2)

        plt.plot(errors1)
        plt.xlabel('Epoch')
        plt.ylabel('Total Error')
        plt.show()

        plt.plot(errors2)
        plt.xlabel('Epoch')
        plt.ylabel('Total Error')
        plt.show()


test_mpl(test_X, test_y)
