# /usr/local/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian, fftconvolve
from time import time
import os
from tqdm import tqdm, trange  # progress bar
import argparse  # to parse arguments in command line
import tensorflow as tf

"""Adapted from the code published from the paper 'Applying Deep Learning 
to Fast Radio Burst Classification' by Liam Connor and Joeri van Leeuwen, as
well as code wrapping done by Vishal Gajjar."""

"""Trains a convolutional neural network to recognize differences between fast
radio bursts and RFI. Training is done by simulating a specified number of FRB
examples and injecting them into noisy Gaussian backgrounds. To include actual
RFI data, psrchive will be used in another file (psr2np.py) to export real data
into numpy formats that this program can inject FRBs into."""

tf.logging.set_verbosity(tf.logging.INFO)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model


class SimulatedFRB(object):
    """ Class to generate a realistic fast radio burst and 
    add the event to data, including scintillation and 
    temporal scattering. @source liamconnor
    """
    def __init__(self, shape=(64, 256), f_ref=1350, bandwidth=1500, max_width=4, tau=0.1):
        assert type(shape) == tuple and len(shape) == 2, "shape needs to be a tuple of 2 integers"        
        self.shape = shape

        # reference frequency (MHz) of observations
        self.f_ref = f_ref
        
        # maximum width of pulse, high point of uniform distribution for pulse width
        self.max_width = max_width 
        
        # number of bins/data points on the time (x) axis
        self.nt = shape[1] 
        
        # frequency range for the pulse, given the number of channels
        self.frequencies = np.linspace(f_ref - bandwidth // 2, f_ref + bandwidth // 2, shape[0])

        # where the pulse will be centered on the time (x) axis
        self.t0 = np.random.randint(-shape[1] + max_width, shape[1] - max_width) 

        # scattering timescale (milliseconds)
        self.tau = tau

        # randomly generated SNR and FRB generated after calling injectFRB()
        self.SNR = None
        self.FRB = None

        '''Simulates background noise similar to the .ar 
        files. Backgrounds will be injected with FRBs to 
        be used in classification later on.'''
        self.background = np.random.randn(*self.shape)

    def gaussian_profile(self):
        """Model pulse as a normalized Gaussian."""
        t = np.linspace(-self.nt // 2, self.nt // 2, self.nt)
        g = np.exp(-(t / np.random.randint(1, self.max_width))**2)
        
        if not np.all(g > 0):
            g += 1e-18

        # clone Gaussian into 2D array with NFREQ rows
        return np.tile(g, (self.shape[0], 1))
    
    def scatter_profile(self):
        """ Include exponential scattering profile."""
        tau_nu = self.tau * (self.frequencies / self.f_ref) ** -4
        t = np.linspace(0, self.nt // 2, self.nt)

        prof = np.exp(-t / tau_nu.reshape(-1, 1)) / tau_nu.reshape(-1, 1)
        return prof / np.max(prof, axis=1).reshape(-1, 1)

    def pulse_profile(self):
        """ Convolve the gaussian and scattering profiles
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile()
        scat_prof = self.scatter_profile()
        
        # convolve the two profiles for each frequency
        # pulse_prof = np.array([fftconvolve(gaus_prof[i], scat_prof[i])[:self.nt] for i in np.arange(self.shape[0])])

        pulse_prof = fftconvolve(gaus_prof, scat_prof, axes=1, mode='same')

        # normalize! high frequencies should have narrower pulses
        pulse_prof /= np.trapz(pulse_prof, axis=1).reshape(-1, 1)
        return pulse_prof

    def scintillate(self):
        """ Include spectral scintillation across the band.
        Approximate effect as a sinusoid, with a random phase
        and a random decorrelation bandwidth.
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()

        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(7)))

        if nscint < 1:
            nscint = 0

        envelope = np.cos(2 * np.pi * nscint * (self.frequencies / self.f_ref)**-2 + scint_phi)
        
        # set all negative elements to zero and add small factor
        envelope[envelope < 0] = 0
        envelope += 0.1

        # add scintillation to pulse profile
        pulse = self.pulse_profile()
        pulse *= envelope.reshape(-1, 1)
        self.FRB = pulse
        return pulse

    def roll(self):
        """Move FRB to random location of the time axis (in-place),
        ensuring that the shift does not cause one end of the FRB
        to end up on the other side of the array."""
        #bin_shift = np.random.randint(low = -self.shape[1] // 2 + self.max_width,
        #                              high = self.shape[1] // 2 - self.max_width)

        bin_shift = np.random.randint(low=0, high = self.shape[1] - self.max_width)
        self.FRB = np.roll(self.FRB, bin_shift, axis=1)

    def fractional_bandwidth(self, frac_low=0.5, frac_high=0.9):
        """Cut some fraction of the full pulse out."""
        # Fraction of frequency (y) axis for the signal
        frac = np.random.uniform(frac_low, frac_high)
        nchan = self.shape[0]

        # collect random fraction of FRB and add to background
        stch = np.random.randint(0, nchan * (1 - frac))
        slice_freq = slice(stch, int(stch + (nchan * frac)))
        slice_FRB = np.copy(self.FRB[slice_freq])
        self.FRB[:, :] = 1e-18
        self.FRB[slice_freq] = slice_FRB

    def sample_SNR(self, SNRmin=8, SNR_sigma=1.0, SNRmax=30):
        """Sample peak SNR from log-normal distribution and throw
        out any value greater than SNRmax."""
        if SNRmin < 0:
            raise ValueError('Minimum SNR cannot be negative')
        if SNRmin > SNRmax:
            raise ValueError('SNRmin cannot be greater than SNRmax')

        random_SNR = SNRmin + np.random.lognormal(mean=1.0, sigma=SNR_sigma)
        if random_SNR < SNRmax:     
            self.SNR = random_SNR
            return random_SNR
        else:
            return self.sample_SNR(SNRmin, SNR_sigma, SNRmax)

    def injectFRB(self, SNR, background=None):
        """Inject the FRB into freq-time array of Gaussian noise"""
        if background is None:
            background = self.background

        # get 1D noise and multiply signal by given SNR
        noise_profile = np.mean(background, axis=0)
        peak_value = SNR * np.std(noise_profile) # originally np.std(noise_profile)
        profile_FRB = np.mean(self.FRB, axis=0)
        
        # make a signal with given SNR
        signal = self.FRB * (peak_value / np.max(profile_FRB))
        return signal

    def simulateFRB(self, background=None, SNRmin=8, SNR_sigma=1.0, SNRmax=15):
        """Combine everything together and inject the FRB into a
        background array of Gaussian noise for the simulation. After
        this method works and is detected by the neural network, proceed
        to inject the FRB into the actual noise files given by psrchive."""
        if background is None:
            background = self.background

        # Create the FRB
        self.scintillate() # make the pulse profile with scintillation
        self.roll() # move the FRB around freq-time array
        self.fractional_bandwidth() # cut out some of the bandwidth
        self.sample_SNR(SNRmin, SNR_sigma, SNRmax) # get random SNR

        # add to the Gaussian noise
        self.simulatedFRB = background + self.injectFRB(background=background, SNR=self.SNR)


def construct_conv2d(train_data, train_labels, eval_data, eval_labels, 
                     nfreq=64, ntime=256, epochs=32, nfilt1=32, nfilt2=64,
                     batch_size=32, saved_model_name='best_model.h5'):
    """ Build a two-dimensional convolutional neural network
    with a binary classifier. Can be used for, e.g.,
    freq-time dynamic spectra of pulsars, dm-time intensity array.

    Parameters:
    ----------
    train_data : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0]=RFI 
    eval_data : ndarray
        (neval, ntime, 1) float64 array with evaluation data
    eval_labels : 
        (neval, 2) binary labels of eval data 
    epochs : int 
        Number of training epochs 
    nfilt1 : int
        Number of neurons in first hidden layer 
    nfilt2 : int 
        Number of neurons in second hidden layer 
    batch_size : int 
        Number of batches for training   
       
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

    # create nfilt1 convolution filters, each of size 5x5
    # max pool and randomly drop some fraction of nodes to limit overfitting
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(64, 256, 1)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second convolutional layer
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten all neurons and run through fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.2))

    # output probabilities of predictions and choose the maximum
    model.add(Dense(2, activation='softmax'))

    # optimize using stochastic gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print("Using batch_size: %d" % batch_size)
    print("Using %d epochs" % epochs)

    cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                        batch_size=32, write_graph=True, write_grads=False,
                                        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None)

    # save best model according to validation accuracy
    best_model_cb = keras.callbacks.ModelCheckpoint(f"{saved_model_name}", monitor='val_acc', verbose=1,
                                                    save_best_only=True)

    model.fit(train_data, train_labels, validation_data=(eval_data, eval_labels),
              batch_size=batch_size, epochs=epochs, callbacks=[cb, best_model_cb])

    score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)
    print(score)

    return model, score


def get_classification_results(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted 
    label (y_pred) for a binary classifier, and return 
    true_positives, false_positives, true_negatives, false_negatives
    """
    true_positives = np.where((y_true == 1) & (y_pred == 1))[0]
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    true_negatives = np.where((y_true == 0) & (y_pred == 0))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    
    return true_positives, false_positives, true_negatives, false_negatives


def confusion_mat(y_true, y_pred):
    """ Generate a confusion matrix for a
    binary classifier based on true labels (
    y_true) and model-predicted label (y_pred)

    returns np.array([[TP, FP],[FN, TN]])
    """
    TP, FP, TN, FN = get_classification_results(y_true, y_pred)

    NTP = len(TP)
    NFP = len(FP)
    NTN = len(TN)
    NFN = len(FN)

    conf_mat = np.array([[NTP, NFP], [NFN, NTN]])
    return conf_mat


def print_metric(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted 
    label (y_pred) for a binary classifier
    and print a confusion matrix, metrics, 
    return accuracy, precision, recall, fscore
    """
    conf_mat = confusion_mat(y_true, y_pred)

    NTP, NFP, NTN, NFN = conf_mat[0, 0], conf_mat[0, 1], conf_mat[1, 1], conf_mat[1, 0]

    print("Confusion matrix:")

    print('\n'.join([''.join(['{:8}'.format(item) for item in row])
                     for row in conf_mat]))

    accuracy = (NTP + NTN) / conf_mat.sum()
    precision = NTP / (NTP + NFP)
    recall = NTP / (NTP + NFN)
    fscore = 2 * precision * recall / (precision + recall)

    print("accuracy: %f" % accuracy)
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("fscore: %f" % fscore)    

    return accuracy, precision, recall, fscore

def make_labels(num_samples, SNRmin, SNRmax=15):
    '''Simulates the background for num_data number of points and appends to ftdata.
    Each iteration will have just noise and an injected FRB, so the label list should
    be populated with just 0 and 1, which will then be shuffled later.'''

    ftdata = []
    labels = []
    values_SNR = []

    for sim in trange(num_samples):
        # create simulation object and add FRB to it
        event = SimulatedFRB()
        event.simulateFRB(background=None, SNRmin=SNRmin, SNR_sigma=1.0, SNRmax=SNRmax)
        
        # put simulated data into ftdata and label it RFI
        ftdata.append(event.background)
        labels.append(0)

        # inject FRB into data and label it true sighting
        ftdata.append(event.simulatedFRB)
        labels.append(1)
        values_SNR.extend([event.SNR, event.SNR])

    return np.array(ftdata), np.array(labels), np.array(values_SNR)

    
def normalize_data(ftdata):
    ftdata = ftdata.reshape(len(ftdata), -1)
    ftdata -= np.median(ftdata, axis=-1)[:, None]
    ftdata /= np.std(ftdata, axis=-1)[:, None]

    return ftdata


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', metavar='num_samples', type=int, default=1000,
                        help='Number of samples to train neural network on')

    parser.add_argument('--nfilt1', type=int, default=32, help='Number of filters in first convolutional layer')
    
    parser.add_argument('--nfilt2', type=int, default=64, help='Number of filters in second convolutional layer')
    
    parser.add_argument('--SNRmin', type=float, default=5.0, help='Minimum SNR for FRB signal')
    
    parser.add_argument('--SNRmax', type=float, default=15.0, help='Maximum SNR of FRB signal')
    
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs to train with')
    
    parser.add_argument('--savemodel', dest='best_model_file', type=str, default='best_model.h5',
                        help='Filename to save best model in')
    
    parser.add_argument('--confmatname', metavar='confusion matrix name', type=str,
                        default='confusion matrix.png', help='Filename to store final confusion matrix in')

    args = parser.parse_args()

    # Read archive files and extract data arrays
    best_model_name = args.best_model_file  # Path and Pattern to find all the .ar files to read and train on
    confusion_matrix_name = args.confmatname

    NFREQ = 64
    NTINT = 256
    DM = 102.4

    # n_sims passed into the interpreter
    ftdata, label, SNRs = make_labels(args.num_samples, args.SNRmin, args.SNRmax)

    Nfl = ftdata.shape[0]
    nfreq = ftdata.shape[1]
    ntime = ftdata.shape[2]

    print(Nfl, nfreq, ntime)
    print(label)

    dshape = ftdata.shape

    # normalize data
    ftdata = ftdata.reshape(len(ftdata), -1)
    ftdata -= np.median(ftdata, axis=-1)[:, None]
    ftdata /= np.std(ftdata, axis=-1)[:, None]

    # zero out nans
    ftdata[ftdata != ftdata] = 0.0
    ftdata = ftdata.reshape(dshape)

    # Get 4D vector for Keras
    ftdata = ftdata[..., None]

    NTRAIN = int(len(label) * 0.5)

    ind = np.arange(Nfl)
    np.random.shuffle(ind)

    # split indices into training and evaluation set
    ind_train = ind[:NTRAIN]
    ind_eval = ind[NTRAIN:]

    train_data_freq, eval_data_freq = ftdata[ind_train], ftdata[ind_eval]

    train_labels, eval_labels = label[ind_train], label[ind_eval]

    # convert to binary matrix
    train_labels_keras = keras.utils.to_categorical(train_labels)
    eval_labels_keras = keras.utils.to_categorical(eval_labels)

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    # Fit convolution neural network to the training data
    model_freq_time, score_freq_time = construct_conv2d(train_data=train_data_freq, train_labels=train_labels_keras,
                                                        eval_data=eval_data_freq, eval_labels=eval_labels_keras,
                                                        epochs=args.epochs, nfilt1=args.nfilt1, nfilt2=args.nfilt2,
                                                        nfreq=NFREQ, ntime=NTINT, saved_model_name=best_model_name)

    y_pred_prob = model_freq_time.predict(eval_data_freq)[:, 1]
    y_pred_freq_time = np.round(y_pred_prob)
    metrics = print_metric(eval_labels, y_pred_freq_time)

    TP, FP, TN, FN = get_classification_results(eval_labels, y_pred_freq_time)

    if TP.size:
        TPind = TP[np.argmin(y_pred_prob[TP])]  # Min probability True positive candidate
        TPdata = eval_data_freq[..., 0][TPind]
    else:
        TPdata = np.zeros((NFREQ, NTINT))

    if FP.size:
        FPind = FP[np.argmax(y_pred_prob[FP])]  # Max probability False positive candidate
        FPdata = eval_data_freq[..., 0][FPind]
    else:
        FPdata = np.zeros((NFREQ, NTINT))

    if FN.size:
        FNind = FN[np.argmax(y_pred_prob[FN])]  # Max probability False negative candidate
        FNdata = eval_data_freq[..., 0][FNind]
    else:
        FNdata = np.zeros((NFREQ, NTINT))

    if TN.size:
        TNind = TN[np.argmin(y_pred_prob[TN])]  # Min probability True negative candidate
        TNdata = eval_data_freq[..., 0][TNind]
    else:
        TNdata = np.zeros((NFREQ, NTINT))

    plt.subplot(221)
    plt.gca().set_title('TP')
    plt.imshow(TPdata, aspect='auto', interpolation='none')
    plt.subplot(222)
    plt.gca().set_title('FP')
    plt.imshow(FPdata, aspect='auto', interpolation='none')
    plt.subplot(223)
    plt.gca().set_title('FN')
    plt.imshow(FNdata, aspect='auto', interpolation='none')
    plt.subplot(224)
    plt.gca().set_title('TN')
    plt.imshow(TNdata, aspect='auto', interpolation='none')

    plt.savefig(confusion_matrix_name)

    plt.show()