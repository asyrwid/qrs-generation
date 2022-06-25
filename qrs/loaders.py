import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field

from filters import denoise_signal
from qrs.transformations import Resample
from qrs.clustering import cluster_qrs_records
from qrs.features import inject_window_features
from qrs.structures import QrsRecord, ClusterQrsRecord

from typing import Any


NOT_QRS_LABELS = ['!', '"', '+', '[', ']', '~']


@dataclass
class SignalLoader:
    df: pd.DataFrame
    lead: int
    signal: np.array
    denoised_signal: np.array
    record_name: str
    smpl_freq: int

    def __len__(self) -> int:
        return self.df.shape[0]

    def get_record(self, idx):
        # return idx-th qrs
        sta = self.df['start'][idx]
        end = self.df['stop'][idx]
        qrs_signal = self.signal[sta:end]
        denoised_qrs_signal = self.denoised_signal[sta:end]

        record = QrsRecord(
            qrs_signal=qrs_signal,
            denoised_qrs_signal=denoised_qrs_signal,
            qrs_features=self.df.iloc[idx]
        )
        return record

    def __getitem__(self, idx):
        record = self.get_record(idx)
        return record

    def prepare_batches(self, n_seconds_window: int, n_samples=200) -> list[QrsRecord]:
        r_samples = self.df['r_sample'].tolist()
        len_signal = len(self.signal)
        n_points_batch = int(n_seconds_window * self.smpl_freq)
        n_batches = int(len_signal/n_points_batch)

        # we wish to resample all the qrs to the same n_samples
        resample = Resample(n_samples)

        batches = []
        for it in range(n_batches):
            left = it * n_points_batch
            right = (it + 1)*n_points_batch

            # determine QRSs indices between 'left' and 'right'
            # and then append corresponidng records to the batch
            batch_records = []
            for it, r_pos in enumerate(r_samples):
                if left <= r_pos < right:
                    record = self.get_record(it)
                    batch_records.append(resample(record))

            batches.append(batch_records)
        return batches

    @property
    def size(self) -> int:
        return len(self)

    def __rich_repr__(self):
        yield 'QRS Signal Loader'
        yield 'records', self.size
        yield 'signal', self.signal.shape[0]
        yield 'record_name', self.record_name
        yield 'lead', self.lead


@dataclass
class ClusterLoader:
    signal_loader: SignalLoader
    clusters: list[ClusterQrsRecord] = field(default_factory=list)
    n_seconds_window: int = 10
    transform: Any = None
    use_denoised_signal: bool = True

    @property
    def lead(self) -> int:
        return self.signal_loader.lead

    def __len__(self) -> int:
        return len(self.clusters)

    def __post_init__(self):
        self.clusters = self.prepare_clusters(
            denoised=self.use_denoised_signal)

    def prepare_clusters(self, denoised=True) -> None:
        clusters = []
        qrs_batches = self.signal_loader.prepare_batches(self.n_seconds_window)
        for qrs_records in qrs_batches:

            # transform qrs within a batch if any transformation is applied
            if self.transform:
                for it in len(qrs_records):
                    qrs_records[it] = self.transform(qrs_records[it])

            local_clusters = cluster_qrs_records(
                qrs_records, denoised=denoised)

            for it in range(len(local_clusters)):
                clusters.append(local_clusters[it])

        return clusters

    def __getitem__(self, idx) -> ClusterQrsRecord:
        return self.clusters[idx]

    @property
    def size(self) -> int:
        return len(self.clusters)

    @property
    def record_name(self) -> str:
        return self.signal_loader.record_name

    def __rich_repr__(self):
        yield 'QRS Cluster Loader'
        yield 'n_clusters', self.size
        yield 'record_name', self.record_name
        yield 'lead', self.lead


def qrs_separation(qrs: pd.DataFrame, signal_length) -> tuple[list[int], list[int]]:
    # This isn't perfect, but let's assume that the signal starts exactly
    # in the middle between two QRSs
    start = [0]
    stop = []
    n_r = len(qrs['r_sample'])

    # We need access to the previous QRS
    # so we have to start with the second one
    for it in range(1, n_r):
        r_idx = qrs['r_sample'][it]
        prev_r_idx = qrs['r_sample'][it - 1]

        # Current QRS begins half way in between the previous R wave and its own
        sta = r_idx - int((r_idx - prev_r_idx)/2)

        # Start of this QRS is the end of the previous one
        # we keep the `start` list is one step ahead of `stop`
        start.append(sta)
        stop.append(sta)
    stop.append(signal_length)
    return start, stop


def load_record(path: str, list_n_seconds: list[int]) -> list[SignalLoader]:
    """
    list_n_seconds list[int]:
        for every value *n_seconds* given we add a set of features
        calculated on a moving window with width *n_seconds*
    """
    # Read the wfdb record
    ann = wfdb.rdann(path, 'atr')
    annotations_frame = pd.DataFrame({
        'label': ann.symbol,
        'r_sample': ann.sample
    })
    record = wfdb.rdrecord(path)
    sampling_freq = record.fs

    # Throw away not-qrs annotations
    qrs_ids = ~annotations_frame.label.isin(NOT_QRS_LABELS)
    qrs = annotations_frame[qrs_ids].reset_index(drop=True)

    # Get all leads available
    signals = record.p_signal.transpose()
    signal_length = len(signals[0])

    # determine boundaries for consecutive qrs signals
    start, stop = qrs_separation(qrs, signal_length)

    n_leads = len(signals)
    signal_loaders = []
    for lead_idx in range(n_leads):
        lead = lead_idx + 1
        # store qrs features
        df = pd.DataFrame({
            'lead': lead,
            'label': np.array(qrs['label']),
            'r_sample': np.array(qrs['r_sample']),
            'r_time': np.array(qrs['r_sample'] / sampling_freq),
            'start': start,
            'stop': stop
        })

        signal = signals[lead_idx]
        denoised_signal = denoise_signal(signal, sampling_freq)

        # store features and information about the source
        signal_loader = SignalLoader(
            df=df,
            lead=lead,
            signal=signal,
            denoised_signal=denoised_signal,
            record_name=ann.record_name,
            smpl_freq=sampling_freq
        )
        # inject addtional features related to n_seconds context window
        for n_seconds in list_n_seconds:
            inject_window_features(signal_loader, n_seconds=n_seconds)

        signal_loaders.append(signal_loader)

    return signal_loaders


def prepare_signal_loaders(mit_paths: list[str], list_n_seconds=[5]) -> list[SignalLoader]:
    signal_loaders = []
    for path in tqdm(mit_paths, desc='Signal loaders'):
        record_signal_loaders = load_record(path, list_n_seconds)
        signal_loaders += record_signal_loaders

    return signal_loaders


def prepare_cluster_loaders(
    mit_paths: list,
    list_n_seconds: list[int] = [5],
    n_seconds_window: int = 10,
    transform: Any = None,
    use_denoised_signal: bool = True
) -> list[ClusterLoader]:
    signal_loaders = prepare_signal_loaders(
        mit_paths=mit_paths,
        list_n_seconds=list_n_seconds
    )
    cluster_loaders = []
    for signal_loader in tqdm(signal_loaders, desc='Cluster loaders'):
        cluster = ClusterLoader(
            signal_loader=signal_loader,
            n_seconds_window=n_seconds_window,
            transform=transform,
            use_denoised_signal=use_denoised_signal
        )
        cluster_loaders.append(cluster)

    return cluster_loaders
