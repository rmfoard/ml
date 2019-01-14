"""Draen from Jason Brownlee's example dense network, from:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""
import numpy
import sys

from keras.models import Sequential
from keras.layers import Dense


# Fix the random seed for reproducibility.
numpy.random.seed(7)

# Load the ruleparts + outcomes data.
dataset = numpy.loadtxt("C1031ruleparts_outcomes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Create the model.
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

sys.exit(0)
