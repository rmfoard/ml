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
dataset = numpy.loadtxt("C1031yitersmap.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,2:114]
Y = dataset[:,1]

# Create the model.
model = Sequential()
model.add(Dense(112, input_dim=112, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear'))

# Set up an optimizer.
opt = optimizers.Adam(lr=1e-3, decay=1e-3 / 200)

# Compile the model.
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

# Fit the model.
#model.fit(X, Y, epochs=150, batch_size=10)
model.fit(X, Y, epochs=1280, batch_size=10)

# Evaluate the model.
scores = model.evaluate(X, Y)
print(scores * 100)
sys.exit(0)
