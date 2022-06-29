import numpy as np
import pandas as pd
from tqdm import tqdm

from qrs import mit
from qrs.loaders import prepare_cluster_loaders, prepare_signal_loaders


def export_datasets():
    export_cluster_dataset(n_per_loader=100)
    export_qrs_dataset(n_per_loader=200)


def export_cluster_dataset(n_per_loader: int = 100):
    paths = mit.mit_paths()
    df = build_cluster_dataset(
        mit_paths=paths,
        n_per_loader=n_per_loader
    )
    filepath = 'tmp/clustered_qrs_dataset.pickle'
    print('Dataset saved to:', filepath)
    df.to_pickle(filepath)


def export_qrs_dataset(n_per_loader: int = 100):
    paths = mit.mit_paths()
    df = build_qrs_dataset(
        mit_paths=paths,
        n_per_loader=n_per_loader
    )
    filepath = 'tmp/raw_qrs_dataset.pickle'
    print('Dataset saved to:', filepath)
    df.to_pickle(filepath)


def build_qrs_dataset(
    mit_paths: list,
    n_per_loader: int = 100,
) -> pd.DataFrame:
    signal_loaders = prepare_signal_loaders(mit_paths)
    pbar = tqdm(enumerate(signal_loaders), total=len(signal_loaders))

    records = []
    for it, signal_loader in pbar:
        # Take just a couple random qrs per signal
        idxs = np.random.choice(
            signal_loader.size,
            size=n_per_loader,
            replace=False
        )
        record_name = signal_loader.record_name

        for idx in idxs:
            qrs_record = signal_loader[idx]

            record = qrs_record.qrs_features.to_dict()
            record['qrs_signal'] = qrs_record.qrs_signal
            record['denoised_qrs_signal'] = qrs_record.denoised_qrs_signal
            record['record_name'] = record_name
            records.append(record)

    df = pd.DataFrame(records)

    return df


def build_cluster_dataset(
    mit_paths: list[str],
    n_per_loader: int = 100
) -> pd.DataFrame:
    cluster_loaders = prepare_cluster_loaders(mit_paths)
    pbar = tqdm(enumerate(cluster_loaders), total=len(cluster_loaders))

    records = []
    for it, cluster_loader in pbar:
        # Take just a couple random clusters per signal
        idxs = np.random.choice(
            cluster_loader.size,
            size=n_per_loader,
            replace=False
        )

        for idx in idxs:
            cluster_record = cluster_loader[idx]
            record_name = cluster_loader.signal_loader.record_name

            # clusters are label-sensitive thus each cluster contains
            # signals characterized by the same label
            label = cluster_record.qrs_records[0].qrs_features.label
            record = {
                'average_signal': cluster_record.average_signal,
                'record_name': record_name,
                'label': label
            }
            records.append(record)

    df = pd.DataFrame(records)

    return df


if __name__ == "__main__":
    export_datasets()
