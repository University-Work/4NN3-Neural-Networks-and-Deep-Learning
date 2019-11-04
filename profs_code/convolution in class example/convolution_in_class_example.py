import numpy

inputArray = numpy.array([[1., 2., 3., 4.],[5., 6., 7., 8.],[9., 10., 11., 12.],[13., 14., 15., 16.]])

kernel = numpy.array([[1., 2.],[3., 4.]])

outputArray = numpy.zeros((3,3))


# slow...
for i in range(3):
    for j in range (3):
        outputArray[i,j] = numpy.sum(numpy.multiply(kernel, inputArray[i:i+2,j:j+2]))

p = numpy.empty((4,0))
for i in range(3):
    for j in range(3):
        p = numpy.append(p,numpy.reshape(inputArray[i:i+2, j:j+2],(4,1)),axis = 1)


outputArray2 = numpy.reshape(numpy.dot(numpy.reshape(kernel,(1,4)),p),(3,3))


print("done")
