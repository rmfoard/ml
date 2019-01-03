import math
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


def genXy(source, sample_len, nr_samples):
    X = []
    y = []
    source_ix = 0

    for sample_nr in range(nr_samples):
        for i in range(sample_len):
            X.append(source[source_ix])
            source_ix = (source_ix + 1) % len(source)
        y.append(source[source_ix])
        source_ix = (source_ix + 1) % len(source)
    return X, y


if __name__ == '__main__':
    source_list = ['a', 'l', 'p', 'h', 'a', 'b', 'e', 't', 'a']
#   source_list = ['2', '4', '6', '8', '10', '12', '14', '16', '18']

    """Prepare for one-hot encoding by first encoding the source strings
    as integers. The encoding is 1:1.
    """
    label_encoder = LabelEncoder()
    values = np.array(source_list)
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    assert(len(integer_encoded) == len(source_list))
    #print('integer_encoded:')
    #print(integer_encoded)
    #print('integer_encoded shape: ' + str(integer_encoded.shape))

    """Do the one-hot encoding."""
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    """The number of features must equal the length of a one-hot
    vector, which is in turn detemined by the number of distinct
    values in 'source_list' (and 'integer_encoded'). We consequently set
    'nr_features' to the length of the first one-hot vector in
    the one-hot encoding of the source (all the one-hots are the
    same length).
    """
    nr_features = len(onehot_encoded[0])
    #print('nr_features: {}'.format(nr_features))
    #print('onehot_encoded:')
    #print(onehot_encoded)
    #print('onehot_encoded shape: ' + str(onehot_encoded.shape))

    """A sample consists of 'sample_len' time steps' worth of data."""
    sample_len = 6
    nr_samples = 500

    """Generate samples and targets by cycling through the encoded source,
    repeatedly grabbing 'sample_len' items as a step-set and the following
    item for its label/target. 'X' is a list of samples/step-sets, each of
    which is a list of 'sample_len' one-hots. 'y' is a list of labels that
    parallels 'X'. Transform the lists to numpy arrays.
    """
    X, y = genXy(onehot_encoded, sample_len, nr_samples)
    X = np.array(X)
    y = np.array(y)

    """Shape the samples as required for the LSTM model."""
    X = X.reshape(nr_samples, sample_len, nr_features)

    """Compile the model. 'sample_len' is the number of time steps in
    each training input, i.e., the length of a step-set. 'nr_features'
    is the length of the one-hots, each of which represents a data item.
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=(sample_len, nr_features)))
    model.add(Dense(nr_features, activation='softmax'))
    model.compile(optimizer='adam', loss='mse')

    """Train."""
    print('Train...')
    model.fit(X, y, batch_size=10, epochs=10, validation_split=0.5, verbose=0)

    print('Predict...')
    predictions = model.predict(X)

    """Show predictions for the first several training step-set inputs. Each
    element of 'predictions' is a vector giving for each source element
    the probability that it's a proper prediction of the element that
    follows the parallel training input step-set.
    """
    for i in range(30):
        """First, show the two-step decoded step-set for which the
        prediction was made.

        The decoder wants a 2D array, which we already have in
        X[i] (indexed by [time_step_nr, onehot_entry]).
        The decoder returns a column vector containing the label-
        encoded number representing the string (in floating point).
        """

        step_set_onehot_decoded = onehot_encoder.inverse_transform(X[i])
        #print('step_set_onehot_decoded:')
        #print(step_set_onehot_decoded)

        """Transform the column vector ( [[v1], [v2], ..., [vn]] ) into a
        1D array ( [v1, v2, ..., vn] ).
        """
        step_set_onehot_decoded = np.ravel(step_set_onehot_decoded)

        """It's necessary to revert the floating point values created by the
        decoder to integers because the label decoder uses them as indices.
        """
        step_set_integer_decoded = label_encoder.inverse_transform(np.array(step_set_onehot_decoded, dtype='int'))
        print('Sample:' + str(step_set_integer_decoded))

        """Next, decode the prediction.
        
        The decoder wants a 2D array. Predictions[i] contains only a single
        [nr_features] 1D array, so we must reshape it into [1, nr_features].
        As above, we transform the column vector into a 1D array using 'ravel'.
        """
        onehot_decoded = onehot_encoder.inverse_transform(predictions[i].reshape(-1, nr_features))
        onehot_decoded = np.ravel(onehot_decoded)
        integer_decoded = label_encoder.inverse_transform(np.array(onehot_decoded, dtype='int'))

        print('Prediction: ' + integer_decoded[0])
