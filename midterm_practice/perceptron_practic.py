import numpy as np
import random

data = np.loadtxt('magic04_space_delimited.txt', delimiter=' ')

training_data_0, training_labels_0 = data[0:499, 0:9], data[0:499, 10]
training_data_1, training_labels_1 = data[12333:12833, 0:9], data[12333:12833, 10]

training_data_attrs = np.concatenate((training_data_0, training_data_1), axis=0)
training_data_class = np.concatenate((training_labels_0, training_labels_1), axis=0)


# Generte Random Weights
random.seed()
w = np.transpose(np.array([random.random()-0.5, random.random()-0.5, random.random()-0.5]))

# Add a column of 1s to the data
ones = np.ones((999, 1))
training_data_attrs_aurg = np.insert(training_data_attrs, 0, ones, axis=0)

print(training_data_attrs_aurg)

