import numpy as np
from tqdm import tqdm, trange

"""Helper functions for training neural network, including
data preprocessing and computing training results.

Credit to Liam Connor: https://github.com/liamconnor/single_pulse_ml"""

def perturb_dm(spec_original, frb_freqtime, stddev=0.005):
    """Shift background RFI and injected FRB by a small, randomly sampled
    percentage of the original DM to match cases when Heimdall would fail
    to reproduce the exact optimal DM. Shift percentage is normally distributed
    around a mean of 0 and a default standard deviation of 0.05% of the original DM."""

    # compute small amount to perturb DM
    dm = spec_original.dm
    # shifted_dm = dm * (1 + np.random.normal(scale=stddev)) NOTE: CHANGE BACK TO THIS
    shifted_dm = dm - np.random.uniform(low=100, high=400)

    # replace original data with injected FRB data
    # disperse FRB data by small amount found above
    spec_original.data = frb_freqtime
    spec_original.dedisperse(shifted_dm, padval='rotate')

    # return the FRB after being dispersed slightly
    return spec_original.data

def scaling_helper(array):
    stddev = np.std(array)

    # subtract median from each row (spectrum) and divide every 2D array by its global stddev
    array -= np.median(array, axis=1).reshape(-1, 1)
    array[:, :] /= stddev

def scale_data(ftdata):
    """Subtract each frequency channel in 3D array by its median and
    divide each array by its global standard deviation. Perform
    this standardization in chunks to avoid a memory overload."""

    for arr in tqdm(ftdata):
        scaling_helper(arr) #rescaled_chunk

def compute_time_series(ftdata, scale=True):
    """Get the 1D time series representations of all signals in ftdata.
    Assumes ftdata is a 3D array of shape (num_samples, nfreq, ntime).

    If scale=True, also subtract each time series by its median and
    divide by its standard deviation."""

    time_series = np.sum(ftdata, axis=1) # sum up frequency channels for all samples

    if scale:
        # standardize each sample to zero median, unit standard deviation
        # better for neural networks to learn in
        medians = np.median(time_series, axis=1).reshape(-1, 1)
        stddev = np.std(time_series, axis=1).reshape(-1, 1)
        time_series = (time_series - medians) / stddev

    return time_series

def get_classification_results(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted
    label (y_pred) for a binary classifier, and return
    true_positives, false_positives, true_negatives, false_negatives
    """
    true_positives = np.where((y_true == 1) & (y_pred >= 0.5))[0]
    false_positives = np.where((y_true == 0) & (y_pred >= 0.5))[0]
    true_negatives = np.where((y_true == 0) & (y_pred < 0.5))[0]
    false_negatives = np.where((y_true == 1) & (y_pred < 0.5))[0]

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