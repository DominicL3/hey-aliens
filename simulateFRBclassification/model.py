import numpy as np
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv1D, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


"""Build a multi-input convolutional neural network for binary classification.
First branch takes in frequency-time data and is run through a 2D CNN, while the
second branch runs the time series of the frequency-time data through a 1D CNN.

Both network outputs are then concatenated and connected to 2 dense layers,
culminating in a softmax layer predicting the probability of classification."""

def construct_conv2d(num_conv_layers=2, num_filters=32):
    """
    Parameters:
    ----------
    num_conv_layers : int
        Number of convolutional layers to implement (MAX 4 due to pooling layers,
        otherwise Keras will throw an error)
    num_filters : int
        Number of filters in first convolutional layer, doubles after each convolutional block.

    Returns
    -------
    cnn_2d : Keras model
        Model to be used on frequency-time data
    """

    cnn_2d = Sequential()

    # create num_filters convolution filters, each of size 3x3
    cnn_2d.add(Conv2D(num_filters, (3, 3), padding='same', input_shape=(None, None, 1)))
    cnn_2d.add(BatchNormalization()) # standardize all inputs to activation function
    cnn_2d.add(Activation('relu'))
    cnn_2d.add(MaxPooling2D(pool_size=(2, 2))) # max pool to reduce the dimensionality

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in np.arange(num_conv_layers - 1):
        num_filters *= 2
        cnn_2d.add(Conv2D(num_filters, (3, 3), padding='same'))
        cnn_2d.add(BatchNormalization())
        cnn_2d.add(Activation('relu'))
        cnn_2d.add(MaxPooling2D(pool_size=(2, 2)))

    # max pool all feature maps
    cnn_2d.add(GlobalMaxPooling2D())

    return cnn_2d

def construct_time_cnn(num_conv_layers=2, num_filters=32):
    """
    Parameters:
    ----------
    num_conv_layers : int
        Number of convolutional layers to implement (MAX 4 due to pooling layers,
        otherwise Keras will throw an error)
    num_filters : int
        Number of filters in first convolutional layer, doubles after each convolutional block.

    Returns
    -------
    time_cnn : Keras model
        Model to be used on time series data of signal
    """

    time_cnn = Sequential()
    num_filters = num_filters // 2 # time series doesn't need as many filters 2D CNN

    # create num_filters convolution filters, each of size 2x2
    # average pool to reduce the dimensionality
    # stride 1 and NO pooling because the signal spike is very short
    time_cnn.add(Conv1D(num_filters, 1, padding='same', input_shape=(None, 1)))
    time_cnn.add(BatchNormalization())
    time_cnn.add(Activation('relu'))

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in np.arange(num_conv_layers - 1):
        num_filters *= 2
        time_cnn.add(Conv1D(num_filters, 2, padding='same'))
        time_cnn.add(BatchNormalization())
        time_cnn.add(Activation('relu'))

    # max pool all feature maps
    time_cnn.add(GlobalMaxPooling1D())

    return time_cnn

def fit_multi_input_model(train_ftdata, train_time_data, train_labels,
                            eval_ftdata, eval_time_data, eval_labels,
                            epochs=32, num_conv_layers=2, num_filters=32,
                            n_dense1=64, n_dense2=32, batch_size=32,
                            weight_FRB=10, saved_model_name='best_model.h5',
                            previous_model_to_train=None):
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
    num_filters : int
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

    if previous_model_to_train is None:
        # construct each individual network
        cnn_2d = construct_conv2d(num_conv_layers=num_conv_layers, num_filters=num_filters)
        time_cnn = construct_time_cnn(num_conv_layers=num_conv_layers, num_filters=num_filters)

        # use output of models as input to final set of layers
        combined_input = concatenate([cnn_2d.output, time_cnn.output])

        # run through two fully connected layers
        fc_1 = Dense(n_dense1, activation='relu')(combined_input)
        dropout_1 = Dropout(0.4)(fc_1)

        fc_2 = Dense(n_dense2, activation='relu')(dropout_1)
        dropout_2 = Dropout(0.3)(fc_2)

        # predict what the label should be
        pred_layer = Dense(1, activation="sigmoid")(dropout_2)

        # final model accepts freq-time data for Conv2D input and
        # 1D time series data for the Conv1D input
        # predictions will output a scalar for each pair of ftdata/time series samples
        model = Model(inputs=[cnn_2d.input, time_cnn.input], outputs=pred_layer)
    else:
        # load in previously saved model to continue training
        print("Loading in previous model: " + previous_model_to_train)
        model = load_model(previous_model_to_train, compile=False)

    # optimize using Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\nBatch size: %d" % batch_size)
    print("Epochs: %d" % epochs)
    print("Neural network will learn %d parameters" % model.count_params())

    # save model with lowest validation loss
    loss_callback = ModelCheckpoint(saved_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    # cut learning rate in half if validation loss doesn't improve in 5 epochs
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # stop training if validation loss doesn't improve after 15 epochs
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    # (1) Fit model using frequency-time training data and time series training data
    # (2) Evaluate on validation set on every epoch and saving model minimizing val_loss
    # (3) Reduce learning rate if val_loss doesn't improve after 5 epochs (bouncing around minimum)
    # (4) Stop training early if val_loss doesn't improve after 15 epochs
    model.fit(x=[train_ftdata, train_time_data], y=train_labels,
                validation_data=([eval_ftdata, eval_time_data], eval_labels),
                class_weight={0: 1, 1: weight_FRB}, batch_size=batch_size, epochs=epochs,
                callbacks=[loss_callback, reduce_lr_callback, early_stop_callback])

    # one last evaluation for the final model (usually not the best)
    score = model.evaluate([eval_ftdata, eval_time_data], eval_labels, batch_size=batch_size)
    print("Final model score: " + str(score))