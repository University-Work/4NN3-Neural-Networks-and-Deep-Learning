import numpy as np

inputArray = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]])
kernel = np.array([[1., 2.], [3., 4.]])
outputArray = np.zeros((3, 3))

# Slow...
# scipy.signal.convolve2d - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
for i in range(3):
    for j in range(3):
        outputArray[i, j] = np.sum(np.multiply(kernel, inputArray[i:i + 2, j:j + 2]))

p = np.empty((4, 0))
for i in range(3):
    for j in range(3):
        p = np.append(p, np.reshape(inputArray[i:i + 2, j:j + 2], (4, 1)), axis=1)

outputArray2 = np.reshape(np.dot(np.reshape(kernel, (1, 4)), p), (3, 3))

print("DONE")
