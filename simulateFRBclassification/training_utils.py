from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

"""Helper functions for training neural network, including
data preprocessing and computing training results."""

def spec2np(spec_npz):
    """Read in previously saved file containing Spectra objects and
    frequencies and retrieve data, placing it all into one 3D numpy array."""
    spectra_list = spec_npz['spectra_data']
    spectra_data = [spec.data for spec in spectra_list]
    return np.array(spectra_data)

def scale_data(ftdata):
    """Subtract each channel in 3D array by its median and
    divide each array by its global standard deviation."""

    medians = np.median(ftdata, axis=-1)[:, :, np.newaxis]
    stddev = np.std(ftdata.reshape(len(ftdata), -1), axis=-1)[:, np.newaxis, np.newaxis]

    scaled_data = (ftdata - medians) / stddev
    return scaled_data

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


def print_metric(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted
    label (y_pred) for a binary classifier
    and print a confusion matrix, metrics,
    return accuracy, precision, recall, fscore
    """

    def confusion_mat(y_true, y_pred):
        """ Generate a confusion matrix for a binary classifier
        based on true labels (y_true) and model-predicted label (y_pred)

        returns np.array([[TP, FP],[FN, TN]])
        """
        TP, FP, TN, FN = get_classification_results(y_true, y_pred)

        NTP = len(TP)
        NFP = len(FP)
        NTN = len(TN)
        NFN = len(FN)

        conf_mat = np.array([[NTP, NFP], [NFN, NTN]])
        return conf_mat

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

    return accuracy, precision, recall, fscore, conf_mat