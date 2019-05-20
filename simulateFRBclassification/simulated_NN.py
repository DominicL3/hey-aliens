#!/usr/bin/env python3

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

"""Adapted from the code published alongside the paper 'Applying Deep Learning 
to Fast Radio Burst Classification' by Liam Connor and Joeri van Leeuwen, as
well as code wrapping done by Vishal Gajjar."""

"""Trains a convolutional neural network to recognize differences between fast
radio bursts and RFI. Training is done by simulating a specified number of FRB
examples and injecting them into noisy Gaussian backgrounds. To include actual
RFI data, psrchive will be used in another file (psr2np.py) to export real data
into numpy formats that this program can inject FRBs into."""

tf.logging.set_verbosity(tf.logging.INFO)

import keras
from sklearn.metrics import recall_score, precision_score, fbeta_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model


class SimulatedFRB(object):
    """ Class to generate a realistic fast radio burst and 
    add the event to data, including scintillation and 
    temporal scattering. @source liamconnor
    """
    def __init__(self, shape=(64, 256), f_low=800, f_high=2000, f_ref=1350, 
                bandwidth=1500, max_width=4, tau=0.1):
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

        # lowest and highest frequencies in which to inject the FRB (default is for GBT)
        self.f_low = f_low
        self.f_high = f_high

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

        scatter = np.exp(-t / tau_nu.reshape(-1, 1)) / tau_nu.reshape(-1, 1)
        
        # normalize the scattering profile and move it to the middle of the array
        scatter /= np.max(scatter, axis=1).reshape(-1, 1)
        scatter = np.roll(scatter, self.shape[1] // 2, axis=1)

        return scatter

    def pulse_profile(self):
        """ Convolve the gaussian and scattering profiles
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile()
        scat_prof = self.scatter_profile()
        
        # convolve the two profiles for each frequency
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

        # Make number of scintils between 0 and 3 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(3)))

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
        bin_shift = np.random.randint(low = -self.shape[1] // 2 + self.max_width,
                                      high = self.shape[1] // 2 - self.max_width)
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

    def normalize_background(self, background):
        """Normalize the background array so each row sums up to 1"""
        background_row_sums = np.trapz(background, axis=1)[:, None]

        # only divide out areas where the row sums up past 0 and isn't nan
        div_cond = np.greater(background_row_sums, 0, out=np.zeros_like(background, dtype=bool), 
                                where=(~np.isnan(background_row_sums))) & (~np.isnan(background))
        
        # normalize background
        normed_background = np.divide(background, background_row_sums, 
                                      out=np.zeros_like(background), 
                                      where=div_cond)

        return normed_background

    def injectFRB(self, SNR, background=None, weights=None):
        """Inject FRB into the background. If specified, signal will 
        be multiplied by the given weights along the frequency axis."""
        if background is None:
            background = self.normalize_background(self.background)
            
        # update the background of the object
        self.background = background

        # remove RFI channels in the background
        if weights is not None:
            if len(weights) != background.shape[0]:
                raise ValueError("Number of input weights does not match number of channels")
            background *= weights.reshape(-1, 1)

        # get 1D noise and multiply signal by given SNR
        noise_profile = np.mean(background, axis=0)
        peak_value = SNR * np.std(noise_profile)
        profile_FRB = np.mean(self.FRB, axis=0)
        
        # make a signal with given SNR
        signal = self.FRB * (peak_value / np.max(profile_FRB))

        # zero out the FRB channels that are low powered on the telescope
        signal[(self.frequencies < self.f_low) | (self.frequencies > self.f_high), :] = 0
        
        # also remove channels from signal that have RFI flagged
        if weights is not None:
            signal *= weights.reshape(-1, 1)

        return background + signal

    def simulateFRB(self, background=None, weights=None, SNRmin=8, SNR_sigma=1.0, SNRmax=15):
        """Combine everything together and inject the FRB into a
        background array (Gaussian noise if background is not specified).
        If given, the signal will be multiplied by the given weights 
        along the frequency axis."""
        if background is None:
            self.normalize_background(self.background)
            background = self.background

        # Create the FRB
        self.scintillate() # make the pulse profile with scintillation
        self.roll() # move the FRB around freq-time array
        self.fractional_bandwidth() # cut out some of the bandwidth
        self.sample_SNR(SNRmin, SNR_sigma, SNRmax) # get random SNR
        
        # add to normalized background
        self.simulatedFRB = self.injectFRB(SNR=self.SNR, background=background, weights=weights)

def construct_conv2d(train_data, train_labels, eval_data, eval_labels, 
                     nfreq=64, ntime=256, epochs=32, n_dense1=256, n_dense2=128,
                     num_conv_layers=4, filter_size=32, batch_size=32,
                     weight_FRB=2, saved_model_name='best_model.h5'):
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

    print("Using batch_size: %d" % batch_size)
    print("Using %d epochs" % epochs)

    class FscoreCallback(keras.callbacks.Callback):
        """Custom metric that will save the model with the highest validation recall as
        training progresses. Will also print out validation precision for good measure."""
        def __init__(self, filepath):
            self.filepath = filepath
            self.epoch = 1
            self.best = -np.inf

        # calculate recall and precision after every epoch
        def on_epoch_end(self, epoch, batch, logs={}):
            if self.epoch > 8: # save active only after certain epoch
                y_pred = np.asarray(self.model.predict(self.validation_data[0]))
                y_pred = np.argmax(y_pred, axis=1)
                
                y_true = self.validation_data[1]
                y_true = np.argmax(y_true, axis=1)
                
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                fscore = fbeta_score(y_true, y_pred, beta=5) # favor recall over precision

                print(f" — val_recall: {recall} — val_precision: {precision} - val_fscore: {fscore}")
                
                if fscore > self.best:
                    print(f'fscore improved from {np.round(self.best, 4)} to {np.round(fscore, 4)}, saving model to {self.filepath}')
                    self.best = recall
                    self.model.save(self.filepath, overwrite=True)
                else:
                    print(f"fscore did not improve from {np.round(self.best, 4)}")

            return

    fscore_callback = FscoreCallback(saved_model_name)

    # save best model according to validation accuracy
    model.fit(x=train_data, y=train_labels, validation_data=(eval_data, eval_labels),
              class_weight={0: 1, 1: weight_FRB}, batch_size=batch_size, epochs=epochs, 
              callbacks=[fscore_callback])

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

def make_labels(num_samples=0, SNRmin=5, SNR_sigma=1.0, SNRmax=15, background_files=None,
                FRB_parameters={'shape': (64, 256), 'f_low': 800, 
                'f_high': 2000, 'f_ref': 1350, 'bandwidth': 1500}):

    '''Simulates the background for num_data number of points and appends to ftdata.
    Each iteration will have just noise and an injected FRB, so the label list should
    be populated with just 0 and 1, which will then be shuffled later.'''

    ftdata = []
    labels = []
    
    if background_files is not None:
        # load in background file and extract data and frequencies
        background_npz = np.load(background_files)
        backgrounds = background_npz['rfi_data']
        freq_RFI = background_npz['freq']
        weights = background_npz['weights']

        # change frequency range of simulated pulse based on incoming RFI files
        FRB_parameters['f_ref'] = np.median(freq_RFI)
        FRB_parameters['bandwidth'] = np.ptp(freq_RFI)

        # set number of samples to iterate over all backgrounds
        num_samples = len(backgrounds)

    # inject FRB into each RFI file or simulate the samples if no backgrounds given
    for sim in trange(num_samples):
        # create simulation object and add FRB to it
        event = SimulatedFRB(**FRB_parameters)
        
        if background_files is None:
            event.simulateFRB(background=None, SNRmin=SNRmin, SNR_sigma=SNR_sigma, SNRmax=SNRmax)
        else:
            # get background and weights from the given array
            background_RFI = backgrounds[sim]
            background_weight = weights[sim]
            
            # inject FRB into real noise array and append label the noise as RFI
            event.simulateFRB(background=background_RFI, weights=background_weight, 
                              SNRmin=SNRmin, SNR_sigma=SNR_sigma, SNRmax=SNRmax)
        
        # append noise to ftdata and label it RFI
        ftdata.append(event.background)
        labels.append(0)

        # inject FRB into data and label it true sighting
        ftdata.append(event.simulatedFRB)
        labels.append(1)

    ftdata, labels = np.array(ftdata), np.array(labels)

    return normalize_data(ftdata), labels

def normalize_data(ftdata):
    """Pretty straightforward, normalizes the data to 
    zero median, unit variance."""
    dshape = ftdata.shape
    
    ftdata = ftdata.reshape(len(ftdata), -1)
    ftdata -= np.median(ftdata, axis=-1)[:, None]
    ftdata /= np.std(ftdata, axis=-1)[:, None]

    # zero out nans
    ftdata[ftdata != ftdata] = 0.0
    ftdata = ftdata.reshape(dshape)

    return ftdata

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--RFI_array', type=str, default=None, help='Array (.npz) that contains RFI data')    

    # parameters that will be used to simulate FRB
    parser.add_argument('f_low', type=float, help='Minimum cutoff frequency (MHz) to inject FRB')
    parser.add_argument('f_high', type=float, help='Maximum cutoff frequency (MHz) to allow inject FRB')
    parser.add_argument('--f_ref', type=float, default=1350, help='Reference frequency (MHz) (center of data)')
    parser.add_argument('--bandwidth', type=float, default=1500, help='Frequency range (MHz) of array')

    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to train neural network on')

    # parameters for convolutional layers
    parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of convolutional layers to train with. Careful when setting this,\
                        the dimensionality of the image is reduced by half with each layer and will error out if there are too many!')
    parser.add_argument('--filter_size', type=int, default=32, 
                        help='Number of filters in starting convolutional layer, doubles with every convolutional block')

    # parameters for dense layers
    parser.add_argument('--n_dense1', type=int, default=128, help='Number of neurons in first dense layer')
    parser.add_argument('--n_dense2', type=int, default=64, help='Number of neurons in second dense layer')
    
    # parameters for signal-to-noise ratio of FRB
    parser.add_argument('--SNRmin', type=float, default=5.0, help='Minimum SNR for FRB signal')
    parser.add_argument('--SNR_sigma', type=float, default=1.0, help='Standard deviation of SNR from log-normal distribution')
    parser.add_argument('--SNRmax', type=float, default=15.0, help='Maximum SNR of FRB signal')

    parser.add_argument('--weight_FRB', type=float, default=10.0, help='Weighting (> 1) on FRBs, used to minimize false negatives')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model training')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs to train with')
    
    # save the model, confusion matrix for last epoch, and validation set
    parser.add_argument('--save_model', dest='best_model_file', type=str, default='best_model.h5',
                        help='Filename to save best model in')
    parser.add_argument('--save_confusion_matrix', dest='confmat', metavar='confusion matrix name', type=str,
                        default='confusion_matrix.png', help='Filename to store final confusion matrix')
    parser.add_argument('--save_classifications', type=str, default='classification_results.npz', 
                        help='Where to save classification results (TP, FP, etc.) and prediction probabilities')

    args = parser.parse_args()

    # Read archive files and extract data arrays
    best_model_name = args.best_model_file  # Path and Pattern to find all the .ar files to read and train on
    confusion_matrix_name = args.confmat
    results_file = args.save_classifications

    # set number of frequency channels to simulate
    if args.RFI_array is not None:
        NFREQ = args.RFI_array['rfi_data'].shape[1]
    else:
        NFREQ = 64
    
    NTIME = 256

    # make dictionaries to pass all the arguments into functions succintly
    frb_params = {'shape': (NFREQ, NTIME), 'f_low': args.f_low, 'f_high': args.f_high,
                  'f_ref': args.f_ref, 'bandwidth': args.bandwidth}
    label_params = {'num_samples': args.num_samples, 'SNRmin': args.SNRmin, 'SNR_sigma': args.SNR_sigma, 
                    'SNRmax': args.SNRmax, 'background_files': args.RFI_array, 'FRB_parameters': frb_params}

    ftdata, labels = make_labels(**label_params)
    
    num_data, nfreq, ntime = ftdata.shape
    print(num_data, nfreq, ntime)
    print(labels)

    # Get 4D vector for Keras
    ftdata = ftdata[..., None]

    NTRAIN = int(len(labels) * 0.5)

    ind = np.arange(num_data)
    np.random.shuffle(ind)

    # split indices into training and evaluation set
    ind_train = ind[:NTRAIN]
    ind_eval = ind[NTRAIN:]

    train_data_freq, eval_data_freq = ftdata[ind_train], ftdata[ind_eval]

    train_labels, eval_labels = labels[ind_train], labels[ind_eval]

    # convert to binary matrix
    train_labels_keras = keras.utils.to_categorical(train_labels)
    eval_labels_keras = keras.utils.to_categorical(eval_labels)

    # used to enable saving the model
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    start_time = time()

    # Fit convolutional neural network to the training data
    model_freq_time, score_freq_time = construct_conv2d(train_data=train_data_freq, train_labels=train_labels_keras,
                                                        eval_data=eval_data_freq, eval_labels=eval_labels_keras,
                                                        nfreq=NFREQ, ntime=NTIME, epochs=args.epochs, batch_size=args.batch_size, 
                                                        num_conv_layers=args.num_conv_layers, filter_size=args.filter_size,
                                                        n_dense1=args.n_dense1, n_dense2=args.n_dense2, 
                                                        weight_FRB=args.weight_FRB, saved_model_name=best_model_name)

    y_pred_prob = model_freq_time.predict(eval_data_freq)[:, 1]
    y_pred_freq_time = np.round(y_pred_prob)
    
    print (f"Training on {args.num_samples} samples took {(time() - start_time) / 60} minutes")
    
    # print out scores of various metrics
    print_metric(eval_labels, y_pred_freq_time)

    TP, FP, TN, FN = get_classification_results(eval_labels, y_pred_freq_time)
    print(f"Saving classification results to {results_file}")
    np.savez(results_file, TP=TP, FP=FP, TN=TN, FN=FN, probabilities=y_pred_prob)

    if TP.size:
        TPind = TP[np.argmin(y_pred_prob[TP])]  # Min probability True positive candidate
        TPdata = eval_data_freq[..., 0][TPind]
    else:
        TPdata = np.zeros((NFREQ, NTIME))

    if FP.size:
        FPind = FP[np.argmax(y_pred_prob[FP])]  # Max probability False positive candidate
        FPdata = eval_data_freq[..., 0][FPind]
    else:
        FPdata = np.zeros((NFREQ, NTIME))

    if FN.size:
        FNind = FN[np.argmax(y_pred_prob[FN])]  # Max probability False negative candidate
        FNdata = eval_data_freq[..., 0][FNind]
    else:
        FNdata = np.zeros((NFREQ, NTIME))

    if TN.size:
        TNind = TN[np.argmin(y_pred_prob[TN])]  # Min probability True negative candidate
        TNdata = eval_data_freq[..., 0][TNind]
    else:
        TNdata = np.zeros((NFREQ, NTIME))

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

    # save data, show plot
    print(f"Saving confusion matrix to {confusion_matrix_name}")
    plt.savefig(confusion_matrix_name)
    plt.show()