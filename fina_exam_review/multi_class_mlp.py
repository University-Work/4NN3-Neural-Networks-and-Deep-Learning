import pandas
import numpy as np
import math
import random
import matplotlib.pyplot as plt

Xraw = np.loadtxt('mfeat.csv', delimiter=',')

    #   ROW COLUMN
X = Xraw[:, 0:6]  # data
maxX = np.max(X, axis=0) # y-axis
X = X/maxX
X = np.concatenate((np.ones((2000, 1)), X), axis=1) # bias column, x-axis
Xc = Xraw[:, 6:16] # labels

random.seed()
rand = random.random()

# inialize the weights - 7 attrbs., 10 possible outputs 

# HIDDEN NEURONS
w1 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5])) # hidden layer 1 - one for each attr., plus bias
w2 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5])) # hidden layer 2
w3 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5, rand-0.5])) # hidden layer 3

# OUTPUT NEURONS
w11 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5])) # output weights, one for each hidden neuron plus bias
w12 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w13 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w14 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w15 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w16 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w17 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w18 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w19 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))
w20 = np.transpose(np.array([rand-0.5, rand-0.5, rand-0.5, rand-0.5]))


# ALGORITHM
alpha = 0.1
epochError = np.zeros(500)
outputError = np.empty((1000, 1))
prediction = np.zeros(10)

# loop over training epochs ( once though the entire training dataset )
for epoch in range(500):
    totalError = 0
    # loop over each datapoint - forward/backward propagation
    for i in range(2000):
        # forward propagation
        z1 = 1/(1 + math.exp(-np.dot(X[i,:], w1)))
        z2 = 1/(1 + math.exp(-np.dot(X[i,:], w2)))
        z3 = 1/(1 + math.exp(-np.dot(X[i,:], w3)))

        xhidden = [1, z1, z2, z3]
        z11 = 1/(1 + math.exp(-np.dot(xhidden, w11)))
        z12 = 1/(1 + math.exp(-np.dot(xhidden, w12)))
        z13 = 1/(1 + math.exp(-np.dot(xhidden, w13)))
        z14 = 1/(1 + math.exp(-np.dot(xhidden, w14)))
        z15 = 1/(1 + math.exp(-np.dot(xhidden, w15)))
        z16 = 1/(1 + math.exp(-np.dot(xhidden, w16)))
        z17 = 1/(1 + math.exp(-np.dot(xhidden, w17)))
        z18 = 1/(1 + math.exp(-np.dot(xhidden, w18)))
        z19 = 1/(1 + math.exp(-np.dot(xhidden, w19)))
        z20 = 1/(1 + math.exp(-np.dot(xhidden, w20)))

        # update the output error
        prediction[0] = z11
        prediction[1] = z12
        prediction[2] = z13
        prediction[3] = z14
        prediction[4] = z15
        prediction[5] = z16
        prediction[6] = z17
        prediction[7] = z18
        prediction[8] = z19
        prediction[9] = z20 

        totalError = totalError + (np.argmax(prediction) != np.argmax(Xc[i, :]))

        # backward propagation
        # update the error
        error11 = z11*(1-z11)*(Xc[i, 0]-z11)
        error12 = z11*(1-z12)*(Xc[i, 0]-z12)
        error13 = z11*(1-z13)*(Xc[i, 0]-z13)
        error14 = z11*(1-z14)*(Xc[i, 0]-z14)
        error15 = z11*(1-z15)*(Xc[i, 0]-z15)
        error16 = z11*(1-z16)*(Xc[i, 0]-z16)
        error17 = z11*(1-z17)*(Xc[i, 0]-z17)
        error18 = z11*(1-z18)*(Xc[i, 0]-z18)
        error19 = z11*(1-z19)*(Xc[i, 0]-z19)
        error20 = z11*(1-z20)*(Xc[i, 0]-z20)

        error1 = z1*(1-z1)*(error11*w11[1] + error12*w12[1] + error13*w13[1] + error14*w14[1] + error15*w15[1] +
                            error16*w16[1] + error17*w17[1] + error18*w18[1] + error19*w19[1] + error20*w20[1])

        error2 = z1*(1-z2)*(error11*w11[2] + error12*w12[2] + error13*w13[2] + error14*w14[2] + error15*w15[2] +
                            error16*w16[2] + error17*w17[2] + error18*w18[2] + error19*w19[2] + error20*w20[2])
        
        error3 = z1*(1-z3)*(error11*w11[3] + error12*w12[3] + error13*w13[3] + error14*w14[3] + error15*w15[3] +
                            error16*w16[3] + error17*w17[3] + error18*w18[3] + error19*w19[3] + error20*w20[3])

        # update the weights
        w1 = w1 + [alpha*error1, alpha*error1*X[i,1], alpha*error1*X[i,2], alpha*error1*X[i,2], 
                    alpha*error1*X[i,4], alpha*error1*X[i,5], alpha*error1*X[i,6]]

        w2 = w2 + [alpha*error2, alpha*error2*X[i,1], alpha*error2*X[i,2], alpha*error2*X[i,2], 
                    alpha*error2*X[i,4], alpha*error2*X[i,5], alpha*error2*X[i,6]]

        w3 = w3 + [alpha*error3, alpha*error3*X[i,1], alpha*error3*X[i,2], alpha*error3*X[i,2], 
                    alpha*error3*X[i,4], alpha*error3*X[i,5], alpha*error3*X[i,6]]

        w11 = w11 + [alpha*error11, alpha*error11*z1, alpha*error11*z2, alpha*error11*z3]
        w12 = w12 + [alpha*error12, alpha*error12*z1, alpha*error12*z2, alpha*error12*z3]
        w13 = w13 + [alpha*error13, alpha*error13*z1, alpha*error13*z2, alpha*error13*z3]
        w14 = w14 + [alpha*error14, alpha*error14*z1, alpha*error14*z2, alpha*error14*z3]
        w15 = w15 + [alpha*error15, alpha*error15*z1, alpha*error15*z2, alpha*error15*z3]
        w16 = w16 + [alpha*error16, alpha*error16*z1, alpha*error16*z2, alpha*error16*z3]
        w17 = w17 + [alpha*error17, alpha*error17*z1, alpha*error17*z2, alpha*error17*z3]
        w18 = w18 + [alpha*error18, alpha*error18*z1, alpha*error18*z2, alpha*error18*z3]
        w19 = w19 + [alpha*error19, alpha*error19*z1, alpha*error19*z2, alpha*error19*z3]
        w20 = w20 + [alpha*error20, alpha*error20*z1, alpha*error20*z2, alpha*error20*z3]

print("Iteration...", epoch+1, "Error = ", totalError, "     ", end = '\r', flush = True)
epochError[epoch] = totalError
