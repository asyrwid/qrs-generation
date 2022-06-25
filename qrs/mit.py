from glob import glob

from qrs.loaders import load_record


def mit_paths() -> list[str]:
    query = 'data/mit-bih-arrhythmia-database-1.0.0/*.dat'
    paths = glob(query)
    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths


class DatasetMIT():
    def __init__(self, paths: str, list_n_seconds=[5], transform=None):
        self.transform = transform
        self.paths = paths

        n_samples = 0
        signal_loaders = []
        for path in paths:
            signal_loaders = load_record(path, list_n_seconds)
            n_leads = len(signal_loaders)

            for lead in range(n_leads):
                n_samples += len(signal_loaders[lead])
                signal_loaders.append(signal_loaders[lead])

        self.signal_loaders = signal_loaders
        self.n_samples = n_samples

    def __len__(self):  # number of samples in each load!
        return self.n_samples

    def determine_position(self, idx: int):
        """
        Find the record and the local index of the qrs we're looking for
        """
        if idx == 0:
            n_record = 0
            local_idx = 0
        else:
            if idx < self.n_samples:
                n_record = -1
                n_sum = 0
                while idx > n_sum:
                    n_record += 1
                    n_sum += len(self.signal_loaders[n_record])
                dif = n_sum - idx
                local_idx = len(self.signal_loaders[n_record]) - dif - 1
            else:
                raise ValueError("index out of range!")

        return n_record, local_idx

    def __getitem__(self, idx: int):
        n_record, local_idx = self.determine_position(idx)
        result = self.signal_loaders[n_record][local_idx]
        if self.transform:
            result = self.transform(result)

        return result
