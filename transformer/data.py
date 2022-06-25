import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset

from ai.transformer import layers as L


class IntSequenceDataset(Dataset):
    def __init__(
        self,
        n_numbers: int = 32,
        sequence_length: int = 10,
        n_samples: int = 16384,
    ):
        if n_numbers <= sequence_length:
            raise ValueError("n_numbers must be greater than sequence_length")
        self.n_samples = n_samples
        self.n_numbers = n_numbers
        self.sequence_length = sequence_length

        # We want the sequence to end on the last available
        # number or sooner
        max_start = n_numbers - sequence_length

        sequence_starts = np.random.randint(0, max_start, size=n_samples)
        sequence_starts = np.tile(sequence_starts, [sequence_length, 1])

        # For each sequence start generate a [0, 1, 2, ... sequence_length - 1] ...
        x = np.linspace(0, sequence_length - 1, sequence_length)
        X = np.tile(x, [n_samples, 1])

        # ... and add it to the starting number
        self.data = sequence_starts.T + X
        self.data = self.data.astype(int)

    @property
    def vocab_size(self) -> int:
        return self.n_numbers

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        row = torch.tensor(self.data[idx], dtype=torch.long)
        x = row[:-1]
        y = row[1:]

        # For just copy the design from the blog
        # even though it's confusing
        sample = {
            "full_sequence": row,
            "x": x,
            "y": y,
        }
        return sample


class ArithmeticSeriesLikeSequenceDataset(Dataset):
    def __init__(self, n_numbers: int = 32, sequence_length: int = 10, n_samples: int = 16384):
        self.n_samples = n_samples
        self.n_numbers = n_numbers
        self.sequence_length = sequence_length

        sequence_starts = np.random.randint(0, n_numbers, size=n_samples)
        sequences = np.tile(sequence_starts, [sequence_length, 1]).T

        # choose randomly steps in the series:
        steps = np.random.randint(1, n_numbers // 2 + 1, size=n_samples)

        for sequence_id in range(n_samples):
            for it in range(1, sequence_length):
                sequences[sequence_id][it] += steps[sequence_id] * it
                cyclic_value = sequences[sequence_id][it] % n_numbers
                sequences[sequence_id][it] = cyclic_value

        # ... and add it to the starting number
        self.data = sequences
        self.data = self.data.astype(int)

    @property
    def vocab_size(self) -> int:
        return self.n_numbers

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        row = torch.tensor(self.data[idx], dtype=torch.long)
        x = row[:-1]
        y = row[1:]

        # For just copy the design from the blog
        # even though it's confusing
        sample = {
            "full_sequence": row,
            "x": x,
            "y": y,
        }
        return sample


class RandomNumbersDataset(Dataset):
    def __init__(self, n_numbers: int = 16, sequence_length: int = 10, n_samples: int = 16384):
        self.n_samples = n_samples
        self.n_numbers = n_numbers
        self.sequence_length = sequence_length

        self.data = np.random.randint(0, self.n_numbers, size=[n_samples, sequence_length])

    @property
    def vocab_size(self) -> int:
        return self.n_numbers

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        row = torch.tensor(self.data[idx], dtype=torch.long)
        x = row[:-1]
        y = row[1:]

        # For just copy the design from the blog
        # even though it's confusing
        sample = {"src": row, "target": x, "target_y": y}
        return sample


def make_src_mask(x: torch.tensor, pad_idx: int = -1) -> torch.tensor:
    src_mask = (x != pad_idx).unsqueeze(-2)
    return src_mask


def make_target_mask(target: torch.tensor, pad_idx: int = -1) -> torch.tensor:
    target_mask = (target != pad_idx).unsqueeze(-2)
    sub_mask = L.subsequent_mask(target.size(-1)).type_as(target_mask.data)
    target_mask = target_mask & Variable(sub_mask)
    return target_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # Encodes the full input sequence into the model
    memory = model.encode(src, src_mask)

    # Prepare a "batch" with the generation prompt
    ys = torch.tensor([[start_symbol]], dtype=src.data.dtype)

    for it in range(max_len - 1):
        target = Variable(ys)
        target_mask = L.subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(
            memory,
            src_mask,
            target,
            Variable(target_mask),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        next_word = next_word.unsqueeze(0)
        ys = torch.cat([ys, next_word], dim=1)
    return ys
