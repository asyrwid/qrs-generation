import numpy as np
from scipy import signal as ss


def bandpass(lowcut: float, highcut: float, fs: float, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = ss.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(
    signal: np.array,
    lowcut: float,
    highcut: float,
    fs: float,
    order=5
) -> np.array:
    b, a = bandpass(lowcut, highcut, fs, order=order)
    y = ss.lfilter(b, a, signal)
    return y


def denoise_signal(signal: np.array, sampling_freq: float, order=2) -> np.array:
    lowcut = .25
    highcut = 25
    denoised = bandpass_filter(
        signal=signal,
        lowcut=lowcut,
        highcut=highcut,
        fs=sampling_freq,
        order=order
    )
    return denoised
