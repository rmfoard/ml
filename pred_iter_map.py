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
dataset = numpy.loadtxt("c1031yiterspart.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,2:10]
Y = dataset[:,1]

# Create the model.
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))

# Set up an optimizer.
sgd = optimizers.SGD(lr=0.1)

# Compile the model.
model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy', 'cosine_proximity'])

# Fit the model.
model.fit(X, Y, epochs=150, batch_size=10)

# Evaluate the model.
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
sys.exit(0)
