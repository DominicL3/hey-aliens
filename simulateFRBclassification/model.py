import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import recall_score, precision_score, fbeta_score

"""Build a two-dimensional convolutional neural network
    with a binary classifier. Can be used for, e.g.,
    freq-time dynamic spectra of pulsars, dm-time intensity array."""

def construct_conv2d(train_data, train_labels, eval_data, eval_labels,
                     nfreq=64, ntime=256, epochs=32, n_dense1=256, n_dense2=128,
                     num_conv_layers=4, filter_size=32, batch_size=32,
                     weight_FRB=2, saved_model_name='best_model.h5'):
    """
    Parameters:
    ----------
    train_data : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0] = RFI
    eval_data : ndarray
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
        Fitted model

    score : np.float
        Accuracy, the fraction of predictions that are correct
    """
    # number of elements for each axis
    nfreq, ntime = train_data.shape[1:3]

    model = Sequential()

    # create filter_size convolution filters, each of size 2x2
    # max pool to reduce the dimensionality
    model.add(Conv2D(filter_size, (2, 2), activation='relu', input_shape=(nfreq, ntime, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in np.arange(num_conv_layers - 1):
        filter_size *= 2
        model.add(Conv2D(filter_size, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten all neurons
    model.add(Flatten())

    # run through two fully connected layers
    model.add(Dense(n_dense1, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(n_dense2, activation='relu'))
    model.add(Dropout(0.3))

    # output prediction probabilities and choose the class with higher probability
    model.add(Dense(2, activation='softmax'))

    # optimize using Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\nBatch size: %d" % batch_size)
    print("Epochs: %d" % epochs)

    class FscoreCallback(keras.callbacks.Callback):
        """Custom metric that will save the model with the highest validation recall as
        training progresses. Will also print out validation precision for good measure."""
        def __init__(self, filepath):
            self.filepath = filepath
            self.epoch = 0
            self.best = -np.inf

        # calculate recall and precision after every epoch
        def on_epoch_end(self, epoch, logs={}):
            self.epoch += 1

            y_pred = np.asarray(self.model.predict(self.validation_data[0]))
            y_pred = np.argmax(y_pred, axis=1)

            y_true = self.validation_data[1]
            y_true = np.argmax(y_true, axis=1)

            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            fscore = fbeta_score(y_true, y_pred, beta=5) # favor recall over precision

            print(" - val_recall: {0} - val_precision: {1} - val_fscore: {2}".format(recall, precision, fscore))

            if epoch > 3:
                if fscore > self.best:
                    print('fscore improved from {0} to {1}, saving model to {2}'.format(np.round(self.best, 4), np.round(fscore, 4), self.filepath))
                    self.best = fscore
                    self.model.save(self.filepath, overwrite=True)
                else:
                    print("fscore ({0}) did not improve from {1}".format(np.round(fscore, 4), np.round(self.best, 4)))
            return

    # save best model according to validation accuracy
    model.fit(x=train_data, y=train_labels, validation_data=(eval_data, eval_labels),
              class_weight={0: 1, 1: weight_FRB}, batch_size=batch_size, epochs=epochs,
              callbacks=[FscoreCallback(saved_model_name)])

    score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)
    print(score)

    return score