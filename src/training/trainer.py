"""Training loop for Word2Vec SGNS."""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tqdm import tqdm

from ..data.dataset import NegativeSampler, SkipGramDataset
from ..model.optimizer import BaseOptimizer, SGD
from ..model.word2vec import Word2Vec


@dataclass
class TrainingConfig:
    """Hyper-parameters for a single training run.

    Attributes
    ----------
    embed_dim : int
        Embedding dimensionality (D).
    window_size : int
        Maximum context window radius.
    num_negatives : int
        Number of negative samples per positive pair (K).
    batch_size : int
        Number of (center, context) pairs per gradient step.
    num_epochs : int
        Number of full passes over the token sequence.
    lr : float
        Initial learning rate.
    min_lr : float
        Final learning rate after linear decay.
    seed : int
        Global random seed for reproducibility.
    eval_interval : int
        Log training loss every ``eval_interval`` batches.
    checkpoint_dir : str | None
        If set, saves model checkpoints here at the end of every epoch.
    """

    embed_dim: int = 100
    window_size: int = 5
    num_negatives: int = 5
    batch_size: int = 512
    num_epochs: int = 3
    lr: float = 0.025
    min_lr: float = 1e-4
    seed: int = 42
    eval_interval: int = 5000
    checkpoint_dir: Optional[str] = None
    loss_history: list[float] = field(default_factory=list)


class Trainer:
    """
    Orchestrates the full SGNS training loop.

    Parameters
    ----------
    model : Word2Vec
        The model whose parameters will be updated.
    dataset : SkipGramDataset
        Source of (center, context) pairs.
    sampler : NegativeSampler
        Produces noise samples for SGNS.
    optimizer : BaseOptimizer
        Gradient update rule (SGD or Adam).
    config : TrainingConfig
    """

    def __init__(
        self,
        model: Word2Vec,
        dataset: SkipGramDataset,
        sampler: NegativeSampler,
        optimizer: BaseOptimizer,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.optimizer = optimizer
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def train(self) -> list[float]:
        """
        Run training for ``config.num_epochs`` epochs.

        Returns a list of per-interval mean losses (for plotting).
        """
        cfg = self.config
        loss_history: list[float] = []

        for epoch in range(1, cfg.num_epochs + 1):
            epoch_losses = self._train_epoch(epoch)
            loss_history.extend(epoch_losses)

            if cfg.checkpoint_dir:
                import os
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"model_epoch{epoch}.npz"
                )
                self.model.save(ckpt_path)

        cfg.loss_history = loss_history
        return loss_history

    def _train_epoch(self, epoch: int) -> list[float]:
        """Train for one pass over the dataset; return interval losses."""
        cfg = self.config
        interval_losses: list[float] = []
        running_loss = 0.0
        batch_count = 0

        batch_iter = self.dataset.batch_iterator(cfg.batch_size, self.rng)
        pbar = tqdm(batch_iter, desc=f"Epoch {epoch}/{cfg.num_epochs}",
                    unit="batch", leave=False, dynamic_ncols=True)

        t0 = time.perf_counter()

        for centers, contexts in pbar:
            negatives = self.sampler.sample_batch(
                centers, contexts, cfg.num_negatives, self.rng
            )
            _, _, loss = self.model.forward(centers, contexts, negatives)

            grad_v_c, grad_u_o, grad_u_n, c_idx, o_idx, n_idx = (
                self.model.backward()
            )
            self.optimizer.step(
                self.model, grad_v_c, grad_u_o, grad_u_n, c_idx, o_idx, n_idx
            )

            running_loss += loss
            batch_count += 1

            if batch_count % cfg.eval_interval == 0:
                avg = running_loss / cfg.eval_interval
                interval_losses.append(avg)
                elapsed = time.perf_counter() - t0
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{self._current_lr:.5f}",
                                 elapsed=f"{elapsed:.0f}s")
                running_loss = 0.0

        # Flush remaining loss
        if running_loss > 0 and batch_count % cfg.eval_interval != 0:
            interval_losses.append(running_loss / (batch_count % cfg.eval_interval))

        return interval_losses

    @property
    def _current_lr(self) -> float:
        if isinstance(self.optimizer, SGD):
            return self.optimizer.current_lr
        return self.config.lr
