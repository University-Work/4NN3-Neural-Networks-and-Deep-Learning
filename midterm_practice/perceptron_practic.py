'''
“A Perceptron in Just a Few Lines of Python Code.” 
MaviccPRP@Web.studio, 29 Mar. 2017, 
https://maviccprp.github.io/a-perceptron-in-just-a-few-lines-of-python-code/.
'''

import numpy as np
import random
import matplotlib.pyplot as plt

data = np.loadtxt('magic04_space_delimited.txt', delimiter=' ')

#    ROW      COLUMN
# [ from:to, from:to ]
X = data[11832:12832, 0:9]
y = data[11832:12832, 10]


def perceptron_sgd_plot(X, y):
    '''
    Train Perceptron & Plot the Total Loss in each Epoch.

    :param X: data samples
    :param y: data labels
    :return: weight vector as a numpy array
    '''

    # Initialize the weight vector for the perceptron with zeros
    w = np.zeros(len(X[0]))
    eta = 1  # Set the learning rate to 1
    epochs = 100  # Set the number of epochs
    errors = []
    total_error = 0

    for t in range(epochs):  # Iterate n times over the whole data set.
        for i, x in enumerate(X):  # 7: Iterate over each sample in the data set
            if(np.dot(X[i], w)*y[i]) <= 0:  # Misclassification condition yi⟨xi,w⟩ ≤ 0
                total_error += (np.dot(X[i], w)*y[i])  # Calculate the Error
                # Update rule for the weights w = w + yi ∗ xi
                w = w = eta*X[i]*y[i]
        errors.append(total_error*(-1))

    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')

    return w


w = perceptron_sgd_plot(X, y)

print(w)
