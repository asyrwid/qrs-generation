from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ai.transformer import data as D


def run_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: Any,
    optimizer: Any = None,
):
    "Standard Training and Logging Function"
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for it, batch in enumerate(data_loader):
        src = batch["src"]
        target = batch["target"]
        target_y = batch["target_y"]
        n_tokens = torch.numel(target_y)

        src_mask = D.make_src_mask(src)
        target_mask = D.make_target_mask(target)
        out = model.forward(
            src,
            target,
            src_mask,
            target_mask,
        )
        y_hat = model.generator(out)
        loss = loss_fn(y_hat, target_y) / n_tokens
        total_loss += loss

        # This is "sequence length"
        total_tokens += n_tokens
        tokens += n_tokens

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / total_tokens


class NoamOpt:
    "Optimizer that implements rate."

    def __init__(
        self,
        optimizer: Any,
        model_size: int,
        factor: int = 1,
        warmup: int = 400,
    ):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        out = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return out

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model):
    adam = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    opt = NoamOpt(
        model.src_embed[0].d_model,
        2,
        4000,
        adam,
    )
    return opt


def make_optimizer(model: nn.Module):
    adam = torch.optim.Adam(
        model.parameters(),
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    opt = NoamOpt(
        model_size=model.src_embed[0].d_model,
        optimizer=adam,
    )
    return opt


class SmoothKLDiv(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx: int = -1, smoothing: float = 0.0):
        super(SmoothKLDiv, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        x = x.contiguous().view(-1, x.size(-1))
        target = target.contiguous().view(-1)

        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # NOTE We can probably remove this, as we're not using
        # padding anyway, but I'm leaving this here in case
        # it helps with debugging
        if self.padding_idx >= 0:
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
