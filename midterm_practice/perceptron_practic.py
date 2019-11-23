'''
“A Perceptron in Just a Few Lines of Python Code.” 
MaviccPRP@Web.studio, 29 Mar. 2017, 
https://maviccprp.github.io/a-perceptron-in-just-a-few-lines-of-python-code/.
'''

import numpy as np
import random
import matplotlib.pyplot as plt

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


def perceptron_sgd_plot(X, y):
    '''
    Train Perceptron & Plot the Total Loss in each Epoch.

        - param X: data samples
        - param y: data labels
        - return: weight vector as a numpy array
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
                w = eta*X[i]*y[i]
        errors.append(total_error*(-1))

    print("Weight Vector: ", w)

    # Plot the Errors
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')
    plt.show()

    return w


perceptron_sgd_plot(X, y)
