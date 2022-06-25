from typing import Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from torch import nn as nn
from torch.utils.data import DataLoader

from ai.transformer import data as D
from ai.transformer import model as M
from ai.transformer import train as T
from ai.transformer import validation as V


@dataclass
class CopyNumbersExperimentConfig:
    # Dataset
    n_numbers: int = 10
    sequence_length: int = 14
    n_training_records: int = 32_000
    n_validation_records: int = 3_000

    # Model
    N: int = 6
    d_ff: int = 2048
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    n_epochs: int = 10
    random_seed: int = 137


@dataclass
class CopyTaskResults:
    model: nn.Module
    train_dataset: D.RandomNumbersDataset
    validation_dataset: D.RandomNumbersDataset
    config: CopyNumbersExperimentConfig
    loss_fn: Any


def make_datasets(config: CopyNumbersExperimentConfig):
    # Train the simple copy task.
    train_dataset = D.RandomNumbersDataset(
        n_numbers=config.n_numbers,
        sequence_length=config.sequence_length,
        n_samples=config.n_training_records,
    )

    validation_dataset = D.RandomNumbersDataset(
        n_numbers=config.n_numbers,
        sequence_length=config.sequence_length,
        n_samples=config.n_validation_records,
    )

    return train_dataset, validation_dataset


def training(config: CopyNumbersExperimentConfig):
    # Reproducibility
    np.random.seed(config.random_seed)

    # Data
    train_dataset, validation_dataset = make_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

    vocab_size = config.n_numbers

    loss_fn = T.SmoothKLDiv(
        size=vocab_size,
        smoothing=0.0,
    )

    model = M.make_model(
        N=config.N,
        src_vocab=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        target_vocab=vocab_size,
    )
    optimizer = T.make_optimizer(model)
    pbar = tqdm(range(config.n_epochs), total=config.n_epochs)
    for epoch in pbar:
        # Training stage
        model.train()
        training_loss = T.run_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
        )

        # Validation stage
        model.eval()
        validation_loss = T.run_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=validation_loader,
        )
        desc = f"Train: {training_loss:.4f}, Test: {validation_loss:.4f}"
        pbar.set_description(desc)

    results = CopyTaskResults(
        model=model,
        config=config,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

    return results


def run():
    """
    For this config I get:
    - Train loss: 0.0018
    - Training time: 30s
    - Wrong token probability: 0.089
    """
    # Train
    config = CopyNumbersExperimentConfig(
        n_numbers=10,
        d_ff=2048,
        N=2,
        n_heads=8,
        d_model=512,
        n_epochs=10,
        sequence_length=10,
        batch_size=30,
        n_training_records=600,
        n_validation_records=100,
    )
    results = training(config)

    review(results)
    return results


def review(experiment_results: CopyTaskResults):
    model = experiment_results.model
    # config = experiment_results.config
    dataset = experiment_results.validation_dataset

    mistake_probability = V.wrong_token_probability(model, dataset)
    print("Wrong token probability:", mistake_probability)
