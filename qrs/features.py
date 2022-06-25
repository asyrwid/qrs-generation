import numpy as np
import pandas as pd


def inject_window_features(signal_loader, n_seconds: int):
    df = signal_loader.df
    fs = signal_loader.smpl_freq
    window_width = int(n_seconds * fs)

    signal = signal_loader.signal
    mean_name = f'mean_{n_seconds}s'
    df[mean_name] = make_window_means(df, signal, window_width)

    denoised_signal = signal_loader.denoised_signal
    denoised_mean_name = f'denoised_mean_{n_seconds}s'
    df[denoised_mean_name] = make_window_means(
        df, denoised_signal, window_width)

    signal_loader.df = df

    return signal_loader


def make_window_means(
    df: pd.DataFrame,
    signal: np.array,
    window_width: int
) -> pd.DataFrame:
    # distance from R in integers corresponding to n_seconds
    # window_width = int(n_seconds * fs)
    r_samples = df['r_sample'].tolist()
    means = []
    for it, r_sample in enumerate(r_samples):
        # determine left and right boundaries for the QRS
        # corresponding to -/+ n_seconds
        left = r_sample - window_width
        right = r_sample + window_width + 1
        if left < 0:
            left = 0
        if right >= len(signal):
            right = len(signal) - 1

        # take indices of QRSs laying in range [left, right]
        qrs_indxs = [r_samples.index(
            r_sample) for r_sample in r_samples if left <= r_sample < right]

        mean = 0.
        for idx in qrs_indxs:
            sta = df.iloc[idx].start
            end = df.iloc[idx].stop
            qrs = signal[sta:end]

            mean += (max(qrs) - min(qrs)) / len(qrs_indxs)

        means.append(mean)

    return means
