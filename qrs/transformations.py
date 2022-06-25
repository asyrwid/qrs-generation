import numpy as np
from scipy.signal import resample

from qrs.structures import QrsRecord


class Normalize(object):
    """ 
    Normalize the qrs signal to a range [0, value]
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, record: QrsRecord):
        qrs_signal = record.qrs_signal
        denoised_qrs_signal = record.denoised_qrs_signal

        qrs_signal -= qrs_signal.min()
        qrs_signal /= qrs_signal.max()
        qrs_signal *= self.value
        record.qrs_signal = qrs_signal

        denoised_qrs_signal -= denoised_qrs_signal.min()
        denoised_qrs_signal /= denoised_qrs_signal.max()
        denoised_qrs_signal *= self.value
        record.denoised_qrs_signal = denoised_qrs_signal

        return record


class NormalizeUnitStd(object):
    """ 
    Normalize the qrs signal so that its mean = 0 and std = 1
    """

    def __call__(self, record: QrsRecord):
        qrs_signal = record.qrs_signal
        denoised_qrs_signal = record.denoised_qrs_signal

        qrs_signal -= np.mean(qrs_signal)
        qrs_signal /= np.std(qrs_signal)
        record.qrs_signal = qrs_signal

        denoised_qrs_signal -= np.mean(denoised_qrs_signal)
        denoised_qrs_signal /= np.std(denoised_qrs_signal)
        record.denoised_qrs_signal = denoised_qrs_signal

        return record


class NormalizeUnitStdWindow(object):
    """ 
    Normalize the qrs signal so that its mean = 0 and std = 1
    on an arbitrary +/-n_seconds window around the R peak
    """

    def __init__(self, n_seconds=5):
        self.n_seconds = n_seconds

    def __call__(self, record: QrsRecord):
        qrs_signal = record.qrs_signal
        denoised_qrs_signal = record.denoised_qrs_signal
        qrs_features = record.qrs_features

        n_sec = str(self.n_seconds)
        mean_name = 'mean_' + n_sec + 's'
        denoised_mean_name = 'denoised_mean_' + n_sec + 's'

        mean = qrs_features[mean_name]
        denoised_mean = qrs_features[denoised_mean_name]

        qrs_signal /= mean
        denoised_qrs_signal /= denoised_mean

        record.qrs_signal = qrs_signal
        record.denoised_qrs_signal = denoised_qrs_signal

        return record


class Resample(object):
    def __init__(self, n_samples=200):
        self.n_samples = n_samples

    def __call__(self, record: QrsRecord):
        record.qrs_signal = resample(record.qrs_signal, self.n_samples)
        record.denoised_qrs_signal = resample(
            record.denoised_qrs_signal, self.n_samples)
        return record
