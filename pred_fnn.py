"""Drawn from Jason Brownlee's example dense network, from:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""
import numpy
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


# Fix the random seed for reproducibility.
numpy.random.seed(7)

# Load the ruleparts + outcomes data.
#dataset = numpy.loadtxt("C1031map_finnrnodes.csv", delimiter=",")
dataset = numpy.loadtxt("toy.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]

# Create the model.
model = Sequential()
model.add(Dense(6, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# Set up an optimizer.
sgd = optimizers.SGD(lr=0.01)

# Compile the model.
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

# Fit the model.
model.fit(X, Y, epochs=150, batch_size=1)

# Evaluate the model.
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
sys.exit(0)
