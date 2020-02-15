#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from time import time
import os, sys
from tqdm import tqdm, trange  # progress bar
import argparse  # to parse arguments in command line
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import load_model

# simulate FRB, create a model, and helper functions for training
from simulate_FRB import SimulatedFRB
from training_utils import *
from model import construct_conv2d

# import waterfaller and filterbank from Vishal's path
sys.path.append('/usr/local/lib/python2.7/dist-packages/')
sys.path.append('/home/vgajjar/linux64_bin/lib/python2.7/site-packages/')

# generate Spectra objects for FRB injection
from waterfaller import filterbank, waterfall
import copy

"""Adapted from the code published alongside the paper 'Applying Deep Learning
to Fast Radio Burst Classification' by Liam Connor and Joeri van Leeuwen, as
well as code wrapping done by Vishal Gajjar."""

"""Trains a convolutional neural network to recognize differences between fast
radio bursts and RFI. Training is done by simulating a specified number of FRB
examples and injecting them into noisy Gaussian backgrounds. To include actual
RFI data, extract_spectra gets real data from filterbank files and turns them
into numpy arrays that this program can inject FRBs into."""

tf.logging.set_verbosity(tf.logging.INFO)

def make_labels(num_samples=0, SNRmin=5, SNR_sigma=1.0, SNRmax=15, background_files=None,
                FRB_parameters={'shape': (64, 256), 'f_low': 800,
                'f_high': 2000, 'f_ref': 1350, 'bandwidth': 1500}):

    """Simulates the background for num_data number of points and appends to ftdata.
    Each iteration will contain one RFI and one FRB array, so the label list should
    be populated with consecutive 0s and 1s, which will then be shuffled later."""

    ftdata = []
    labels = []

    if background_files is not None:
        # extract data from background files
        backgrounds = spec2np(background_files)
        freq_RFI = background_files['freq']

        # change frequency range of simulated pulse based on incoming RFI files
        FRB_parameters['f_ref'] = np.median(freq_RFI)
        FRB_parameters['bandwidth'] = np.ptp(freq_RFI)

        # set number of samples to iterate over all backgrounds
        num_samples = len(backgrounds)

    # inject FRB into each RFI file or simulate the samples if no backgrounds given
    for sim in trange(num_samples):
        event = SimulatedFRB(**FRB_parameters)

        if background_files is None:
            event.simulateFRB(background=None, SNRmin=SNRmin, SNR_sigma=SNR_sigma, SNRmax=SNRmax)
        else:
            # get background and weights from the given array
            background_RFI = backgrounds[sim]

            # inject FRB into real noise array and append label the noise as RFI
            event.simulateFRB(background=background_RFI, SNRmin=SNRmin,
                              SNR_sigma=SNR_sigma, SNRmax=SNRmax)

        # append noise to ftdata and label it RFI
        ftdata.append(event.background)
        labels.append(0)

        # inject FRB into data and label it true sighting
        ftdata.append(event.simulatedFRB)
        labels.append(1)

    ftdata, labels = np.array(ftdata), np.array(labels)

    return ftdata, labels

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()

    # parameters that will be used to simulate FRB
    parser.add_argument('f_low', type=float, help='Minimum cutoff frequency (MHz) to inject FRB')
    parser.add_argument('f_high', type=float, help='Maximum cutoff frequency (MHz) to allow inject FRB')
    parser.add_argument('--f_ref', type=float, default=1350, help='Reference frequency (MHz) (center of data)')
    parser.add_argument('--bandwidth', type=float, default=1500, help='Frequency range (MHz) of array')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to train neural network on.\
                                                                       Only valid if generating Gaussian noise; overwritten\
                                                                       if background files are provided')

    parser.add_argument('--save_spectra', type=str, default=None, help='Filename to save Spectra objects with injected FRBs')

    # option to input RFI array
    parser.add_argument('--RFI_samples', type=str, default=None, help='Array (.npz) that contains RFI data')

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
    parser.add_argument('--save_model', dest='best_model_file', type=str, default='./models/best_model.h5',
                        help='Filename to save best model in')
    parser.add_argument('--save_confusion_matrix', dest='conf_mat', metavar='confusion matrix name', type=str,
                        default='./confusion_matrices/confusion_matrix.png', help='Filename to store final confusion matrix')
    parser.add_argument('--save_classifications', type=str, default=None,
                        help='Where to save classification results (TP, FP, etc.) and prediction probabilities')

    args = parser.parse_args()

    # Read archive files and extract data arrays
    best_model_name = args.best_model_file  # Path and Pattern to find all the .ar files to read and train on
    confusion_matrix_name = args.conf_mat
    results_file = args.save_classifications
    RFI_samples = np.load(args.RFI_samples, allow_pickle=True)

    # set number of frequency channels to simulate
    if RFI_samples is not None:
        print('Getting number of channels from inputted RFI array')
        spectra_samples = RFI_samples['spectra_data']
        NFREQ = spectra_samples[0].numchans
    else:
        NFREQ = 64

    print('Number of frequency channels: {}'.format(NFREQ))
    NTIME = 256

    # make dictionaries to pass all the arguments into functions succintly
    frb_params = {'shape': (NFREQ, NTIME), 'f_low': args.f_low, 'f_high': args.f_high,
                  'f_ref': args.f_ref, 'bandwidth': args.bandwidth}
    label_params = {'num_samples': args.num_samples, 'SNRmin': args.SNRmin, 'SNR_sigma': args.SNR_sigma,
                    'SNRmax': args.SNRmax, 'background_files': RFI_samples, 'FRB_parameters': frb_params}

    ftdata, labels = make_labels(**label_params)

    # save spectra with matching labels
    if args.save_spectra is not None:
        print('Saving 10000 spectra to disk as  ' + args.save_spectra)
        # make two copies of spectra to "recreate all labeled Spectra objects
        spectra1 = copy.deepcopy(RFI_samples['spectra_data'])
        spectra2 = copy.deepcopy(RFI_samples['spectra_data'])
        spectra = np.insert(spectra1, np.arange(len(spectra1)), spectra2)

        assert len(spectra) == len(labels), "Not the same shape"

        # get a bunch of spectra and labels for simulated arrays
        rand_idx = np.random.randint(0, len(spectra), size=10000)
        random_spectra = spectra[rand_idx]

        # replace spectra data with itself or simulated FRB
        random_data, random_labels = ftdata[rand_idx], labels[rand_idx]

        # random_spectra, random_data, random_labels = spectra[1::2], ftdata[1::2], labels[1::2]

        for spec, data in zip(random_spectra, random_data):
            spec.data = data

        np.savez(args.save_spectra, spectra=random_spectra, labels=random_labels)

    # bring each channel to zero median and each array to unit stddev
    print('Scaling arrays. . .')
    ftdata = scale_data(ftdata)
    print('Done scaling!')

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

    # encode RFI as [1, 0] and FRB as [0, 1]
    train_labels_keras = to_categorical(train_labels)
    eval_labels_keras = to_categorical(eval_labels)

    # used to enable saving the model
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    start_time = time()

    # Fit convolutional neural network to the training data
    score = construct_conv2d(train_data=train_data_freq, train_labels=train_labels_keras,
                            eval_data=eval_data_freq, eval_labels=eval_labels_keras,
                            nfreq=NFREQ, ntime=NTIME, epochs=args.epochs, batch_size=args.batch_size,
                            num_conv_layers=args.num_conv_layers, filter_size=args.filter_size,
                            n_dense1=args.n_dense1, n_dense2=args.n_dense2,
                            weight_FRB=args.weight_FRB, saved_model_name=best_model_name)

    # load the best model saved to test out confusion matrix
    model_freq_time = load_model(best_model_name, compile=True)
    y_pred_prob = model_freq_time.predict(eval_data_freq)[:, 1]
    y_pred_freq_time = np.round(y_pred_prob)

    print("Training on {0} samples took {1} minutes".format(len(train_labels), np.round((time() - start_time) / 60, 2)))

    # print out scores of various metrics
    accuracy, precision, recall, fscore, conf_mat = print_metric(eval_labels, y_pred_freq_time)

    TP, FP, TN, FN = get_classification_results(eval_labels, y_pred_freq_time)

    if results_file is not None:
        print("Saving classification results to {0}".format(results_file))
        np.savez(results_file, TP=TP, FP=FP, TN=TN, FN=FN, probabilities=y_pred_prob)

    # get lowest confidence selection for each category
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

    # plot the confusion matrix and display
    plt.subplot(221)
    plt.gca().set_title('TP: {}'.format(conf_mat[0][0]))
    plt.imshow(TPdata, aspect='auto', interpolation='none')
    plt.subplot(222)
    plt.gca().set_title('FP: {}'.format(conf_mat[0][1]))
    plt.imshow(FPdata, aspect='auto', interpolation='none')
    plt.subplot(223)
    plt.gca().set_title('FN: {}'.format(conf_mat[1][0]))
    plt.imshow(FNdata, aspect='auto', interpolation='none')
    plt.subplot(224)
    plt.gca().set_title('TN: {}'.format(conf_mat[1][1]))
    plt.imshow(TNdata, aspect='auto', interpolation='none')
    plt.tight_layout()

    # save data, show plot
    print("Saving confusion matrix to {}".format(confusion_matrix_name))
    plt.savefig(confusion_matrix_name, dpi=300)
    plt.show()
