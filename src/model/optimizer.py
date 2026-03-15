"""
Gradient-based optimizers for sparse embedding updates.

Both optimizers operate on *sparse* index sets — only the rows of W and W_ctx
that appear in the current batch are updated.  This is critical for efficiency:
updating the full (V, D) matrix on every step would dominate wall-clock time.

Implemented optimizers
-----------------------
SGD  — Stochastic Gradient Descent with optional learning-rate decay
Adam — Adaptive Moment Estimation (Kingma & Ba, 2015) with sparse support
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """Common interface for all optimizers."""

    @abstractmethod
    def step(
        self,
        model: "Word2Vec",  # type: ignore[name-defined]  # imported at runtime
        grad_v_c: np.ndarray,
        grad_u_o: np.ndarray,
        grad_u_n: np.ndarray,
        centers: np.ndarray,
        contexts: np.ndarray,
        negatives: np.ndarray,
    ) -> None:
        """Apply one gradient-descent update step."""


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent with linear learning-rate decay.

    The learning rate is decayed linearly from ``lr`` to ``min_lr`` over
    ``total_steps`` steps.  Constant learning rate is the special case where
    ``min_lr`` equals ``lr``.

    Parameters
    ----------
    lr : float
        Initial learning rate (default 0.025, same as original word2vec).
    min_lr : float
        Minimum learning rate after full decay.
    total_steps : int
        Total number of optimizer steps for decay schedule.
    """

    def __init__(
        self,
        lr: float = 0.025,
        min_lr: float = 1e-4,
        total_steps: int = 1,
    ) -> None:
        self.lr = lr
        self.min_lr = min_lr
        self.total_steps = max(total_steps, 1)
        self._step = 0

    @property
    def current_lr(self) -> float:
        frac = min(self._step / self.total_steps, 1.0)
        return max(self.lr * (1.0 - frac) + self.min_lr * frac, self.min_lr)

    def step(self, model, grad_v_c, grad_u_o, grad_u_n,
             centers, contexts, negatives) -> None:
        lr = self.current_lr

        # Centre embedding update — accumulate gradients per unique index
        np.add.at(model.W, centers, -lr * grad_v_c)

        # Positive context update
        np.add.at(model.W_ctx, contexts, -lr * grad_u_o)

        # Negative context update — negatives has shape (B, K)
        B, K = negatives.shape
        flat_neg = negatives.reshape(-1)                    # (B*K,)
        flat_grad = grad_u_n.reshape(-1, model.embed_dim)  # (B*K, D)
        np.add.at(model.W_ctx, flat_neg, -lr * flat_grad)

        self._step += 1


class Adam(BaseOptimizer):
    """
    Adam optimizer with sparse moment updates.

    Only rows that appear in the current mini-batch have their first- and
    second-moment estimates updated, reducing both memory bandwidth and the
    bias introduced by updating moments for rows that received zero gradient.

    This is equivalent to the "lazy Adam" or "sparse Adam" variant used in
    many embedding training frameworks.

    Parameters
    ----------
    lr : float
        Step size (default 1e-3).
    beta1 : float
        Exponential decay rate for first moment (default 0.9).
    beta2 : float
        Exponential decay rate for second moment (default 0.999).
    eps : float
        Small constant for numerical stability (default 1e-8).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._t = 0  # global step counter
        self._m_W: np.ndarray | None = None     # 1st moment for W
        self._v_W: np.ndarray | None = None     # 2nd moment for W
        self._m_Wc: np.ndarray | None = None    # 1st moment for W_ctx
        self._v_Wc: np.ndarray | None = None    # 2nd moment for W_ctx

    def _init_moments(self, model) -> None:
        if self._m_W is None:
            shape = (model.vocab_size, model.embed_dim)
            self._m_W = np.zeros(shape, dtype=np.float32)
            self._v_W = np.zeros(shape, dtype=np.float32)
            self._m_Wc = np.zeros(shape, dtype=np.float32)
            self._v_Wc = np.zeros(shape, dtype=np.float32)

    def _sparse_update(
        self,
        param: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        indices: np.ndarray,
        grad: np.ndarray,
        t: int,
    ) -> None:
        """Apply Adam update to selected rows of ``param`` in-place."""
        b1, b2, eps = self.beta1, self.beta2, self.eps
        unique_idx = np.unique(indices)

        for ui in unique_idx:
            mask = indices == ui
            g = grad[mask].mean(axis=0)  # average duplicate-index grads
            m[ui] = b1 * m[ui] + (1 - b1) * g
            v[ui] = b2 * v[ui] + (1 - b2) * g * g
            m_hat = m[ui] / (1 - b1 ** t)
            v_hat = v[ui] / (1 - b2 ** t)
            param[ui] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def step(self, model, grad_v_c, grad_u_o, grad_u_n,
             centers, contexts, negatives) -> None:
        self._init_moments(model)
        self._t += 1
        t = self._t

        self._sparse_update(model.W, self._m_W, self._v_W,
                            centers, grad_v_c, t)
        self._sparse_update(model.W_ctx, self._m_Wc, self._v_Wc,
                            contexts, grad_u_o, t)

        B, K = negatives.shape
        flat_neg = negatives.reshape(-1)
        flat_grad = grad_u_n.reshape(-1, model.embed_dim)
        self._sparse_update(model.W_ctx, self._m_Wc, self._v_Wc,
                            flat_neg, flat_grad, t)
