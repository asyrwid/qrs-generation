import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class QrsRecord:
    qrs_signal: np.array
    denoised_qrs_signal: np.array
    qrs_features: pd.Series

    @property
    def initial_n_samples(self) -> int:
        return self.qrs_features['stop'] - self.qrs_features['start']

    @property
    def current_n_samples(self) -> int:
        return len(self.qrs_signal)


@dataclass
class ClusterQrsRecord:
    qrs_records: list
    average_signal: np.array

    @property
    def size(self) -> int:
        return len(self.qrs_records)

    def __rich_repr__(self) -> None:
        yield 'QRS Cluster Record'
        yield 'n_records', self.size
