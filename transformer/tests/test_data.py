import ai.transformer.data as D


def test_IntSequenceDataset():
    n_numbers = 10
    sequence_length = 3
    n_samples = 5
    dataset = D.IntSequenceDataset(
        n_numbers=n_numbers,
        sequence_length=sequence_length,
        n_samples=n_samples,
    )
    assert list(dataset.data.shape) == [n_samples, sequence_length]
