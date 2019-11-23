import numpy as np
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


def perceptron(X, y):
    # w = np.transpose(np.zeros(len(X[0])))
    random.seed()
    w = np.transpose(np.array([
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5, random.random()-0.5, random.random()-0.5,
        random.random()-0.5
    ]))
    epoch = 1000
    eta = 0.1
    totalErrorPrev = 0

    for epoch in range(epoch):
        totalError = 0
        for i in range(len(X)):
            z = (np.sign(np.dot(X[i, :], w)) + 1)/2  # prediction
            error = y[i] - z
            totalError = totalError + abs(error)
            w = w + eta*error*X[i, :]

            if abs(totalError - totalErrorPrev) < 1:
                # print("Error: ", totalError)
                # print("Epoch", epoch, " i ", i)
                break
            totalErrorPrev = totalError
            # print("Error: ", totalError)
            # print("Epoch", epoch, " i ", i)

    print("Error: ", totalError)
    print("Weight Vector: ", w)

    return w


def test_perceprtron(test_X, test_y):
    errors = []
    w = perceptron(X, y)
    for i in range(len(test_X)):
        prediction = np.dot(test_X[i, :], w)
        error = test_y[i] - prediction
        errors.append(error)

        plt.plot(errors)
        plt.xlabel('Epoch')
        plt.ylabel('Total Error')
        plt.show()


test_perceprtron(test_X, test_y)
