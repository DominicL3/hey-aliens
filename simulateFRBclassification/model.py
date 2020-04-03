import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv1D, AveragePooling1D, Conv2D, AveragePooling2D

from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint, TensorBoard


"""Build a multi-input convolutional neural network for binary classification.
First branch takes in frequency-time data and is run through a 2D CNN, while the
second branch runs the time series of the frequency-time data through a 1D CNN.

Both network outputs are then concatenated and connected to 2 dense layers,
culminating in a softmax layer predicting the probability of classification."""

def construct_conv2d(nfreq, ntime, num_conv_layers=2, filter_size=32):
    """
    Parameters:
    ----------
    nfreq : int
        Number of frequency channels in frequency-time data.
    ntime : int
        Number of time bins in frequency-time data.
    num_conv_layers : int
        Number of convolutional layers to implement (MAX 4 due to pooling layers,
        otherwise Keras will throw an error)
    filter_size : int
        Number of filters in first convolutional layer, doubles after each convolutional block.

    Returns
    -------
    cnn_2d : Keras model
        Model to be used on frequency-time data
    """

    cnn_2d = Sequential()

    # create filter_size convolution filters, each of size 2x2
    # max pool to reduce the dimensionality
    cnn_2d.add(Conv2D(filter_size, (2, 2), activation='relu', input_shape=(nfreq, ntime, 1)))
    cnn_2d.add(AveragePooling2D(pool_size=(2, 2)))

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in np.arange(num_conv_layers - 1):
        filter_size *= 2
        cnn_2d.add(Conv2D(filter_size, (2, 2), activation='relu'))
        cnn_2d.add(AveragePooling2D(pool_size=(2, 2)))

    # flatten all neurons
    cnn_2d.add(Flatten())

    return cnn_2d

def construct_time_cnn(ntime, num_conv_layers=2, filter_size=32):
    """
    Parameters:
    ----------
    ntime : int
        Number of time bins in time series data.
    num_conv_layers : int
        Number of convolutional layers to implement (MAX 4 due to pooling layers,
        otherwise Keras will throw an error)
    filter_size : int
        Number of filters in first convolutional layer, doubles after each convolutional block.

    Returns
    -------
    time_cnn : Keras model
        Model to be used on time series data of signal
    """

    time_cnn = Sequential()

    # create filter_size convolution filters, each of size 2x2
    # average pool to reduce the dimensionality
    time_cnn.add(Conv1D(filter_size, 2, activation='relu', input_shape=(ntime, 1)))
    time_cnn.add(AveragePooling1D(pool_size=2))

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in np.arange(num_conv_layers - 1):
        filter_size *= 2
        time_cnn.add(Conv1D(filter_size, 2, activation='relu'))
        time_cnn.add(AveragePooling1D(pool_size=2))

    # flatten all neurons
    time_cnn.add(Flatten())

    return time_cnn

def fit_multi_input_model(train_ftdata, train_time_data, train_labels,
                            eval_ftdata, eval_time_data, eval_labels,
                            nfreq=64, ntime=256, epochs=32,
                            num_conv_layers=2, filter_size=32,
                            n_dense1=64, n_dense2=32, batch_size=32,
                            weight_FRB=2, saved_model_name='best_model.h5'):
    """
    Parameters:
    ----------
    train_ftdata : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0] = RFI
    eval_ftdata : ndarray
        (neval, ntime, 1) float64 array with evaluation data
    eval_labels :
        (neval, 2) binary labels of eval data
    epochs : int
        Number of training epochs
    num_conv_layers : int
        Number of convolutional layers to implement (MAX 4 due to max pooling layers,
        otherwise Keras will throw an error)
    filter_size : int
        Number of filters in first convolutional layer, doubles after each convolutional block.
    n_dense1 : int
        Number of neurons in first hidden layer
    n_dense2 : int
        Number of neurons in second hidden layer

    batch_size : int
        Number of batches for training
    weight_FRB : float
        Class weight given to FRB during fitting. This means the loss function
        will penalize missing an FRB more with larger weight_FRB.

    Returns
    -------
    model : Keras model
        Model to be used with frequency-time and time series data.

    score : np.float
        Accuracy, the fraction of predictions that are correct
    """

    # construct each individual network
    cnn_2d = construct_conv2d(nfreq, ntime, num_conv_layers=num_conv_layers, filter_size=filter_size)
    time_cnn = construct_time_cnn(ntime, num_conv_layers=num_conv_layers, filter_size=filter_size)

    # use output of models as input to final set of layers
    combined_input = concatenate([cnn_2d.output, time_cnn.output])

    # run through two fully connected layers
    fc_1 = Dense(n_dense1, activation='relu')(combined_input)
    dropout_1 = Dropout(0.4)(fc_1)

    fc_2 = Dense(n_dense2, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.3)(fc_2)

    # predict what the should be
    pred_layer = Dense(2, activation="softmax")(dropout_2)

    # final model accepts freq-time data for Conv2D input and
    # 1D time series data for the Conv1D input
    # predictions will output a scalar for each pair of ftdata/time series samples

    model = Model(inputs=[cnn_2d.input, time_cnn.input], outputs=pred_layer)

    # optimize using Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\nBatch size: %d" % batch_size)
    print("Epochs: %d" % epochs)

    loss_callback = ModelCheckpoint(saved_model_name, monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard_cb = TensorBoard(histogram_freq=0, write_graph=False)

    # fit model using frequency-time training data and
    # time series training data, and evaluate on validation set
    # on every epoch, saving best model according to validation accuracy
    model.fit(x=[train_ftdata, train_time_data], y=train_labels,
                validation_data=([eval_ftdata, eval_time_data], eval_labels),
                class_weight={0: 1, 1: weight_FRB}, batch_size=batch_size,
                epochs=epochs, callbacks=[loss_callback, tensorboard_cb])

    # one last evaluation for the final model (usually not the best)
    score = model.evaluate([eval_ftdata, eval_time_data], eval_labels, batch_size=batch_size)
    print(score)