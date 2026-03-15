"""
Numerical gradient checks for the Word2Vec forward/backward pass.

These tests verify that the analytical gradients derived by ``backward()``
match numerical approximations computed via central differences:

    ∂L/∂θ_i ≈ [L(θ + ε·e_i) − L(θ − ε·e_i)] / (2ε)

A relative tolerance of 1e-4 is used, which is standard for float32
arithmetic.  The tests use tiny models and small batches to keep runtime low.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.word2vec import Word2Vec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_gradient(
    param: np.ndarray,
    row: int,
    col: int,
    loss_fn,
    eps: float = 1e-4,
) -> float:
    """
    Central-difference estimate of ∂loss/∂param[row, col].

    The array is modified in-place and restored after each probe.
    """
    original = float(param[row, col])

    param[row, col] = original + eps
    loss_plus = loss_fn()

    param[row, col] = original - eps
    loss_minus = loss_fn()

    param[row, col] = original  # restore
    return (loss_plus - loss_minus) / (2 * eps)


def make_model_and_batch(
    vocab_size: int = 20,
    embed_dim: int = 8,
    batch_size: int = 4,
    num_negatives: int = 3,
    seed: int = 0,
) -> tuple:
    """Return a tiny model and a fixed random batch."""
    rng = np.random.default_rng(seed)
    model = Word2Vec(vocab_size, embed_dim, seed=seed)

    # Use float64 for the gradient check to reduce rounding errors
    model.W = model.W.astype(np.float64)
    model.W_ctx = model.W_ctx.astype(np.float64)

    # Random valid indices
    centers = rng.integers(0, vocab_size, size=batch_size).astype(np.int32)
    contexts = rng.integers(0, vocab_size, size=batch_size).astype(np.int32)
    negatives = rng.integers(0, vocab_size,
                             size=(batch_size, num_negatives)).astype(np.int32)

    return model, centers, contexts, negatives


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGradients:
    """Analytical vs numerical gradient checks."""

    def _run_check(
        self,
        param: np.ndarray,
        param_name: str,
        analytical_grad: np.ndarray,
        index_rows: np.ndarray,
        model,
        centers,
        contexts,
        negatives,
        eps: float = 1e-4,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        max_checks: int = 10,
    ) -> None:
        """
        Check a sample of entries in ``param`` against numerical gradients.

        ``index_rows`` tells which rows of ``param`` received a gradient;
        we probe one entry per unique row.
        """
        checked = 0
        for row in np.unique(index_rows):
            col = 0  # check the first dimension of each updated row
            row = int(row)

            def loss_fn():
                _, _, loss = model.forward(centers, contexts, negatives)
                return loss

            num_grad = numerical_gradient(param, row, col, loss_fn, eps)

            # Accumulate analytical gradients for this row
            mask = index_rows == row
            analytic = float(analytical_grad[mask].sum(axis=0)[col])

            assert np.isclose(analytic, num_grad, rtol=rtol, atol=atol), (
                f"Gradient mismatch in {param_name}[{row},{col}]: "
                f"analytical={analytic:.6f}, numerical={num_grad:.6f}"
            )
            checked += 1
            if checked >= max_checks:
                break

    def test_grad_center_embeddings(self):
        """∂L/∂v_c should match numerical gradient for W (center embeddings)."""
        model, centers, contexts, negatives = make_model_and_batch()
        model.forward(centers, contexts, negatives)
        grad_v_c, _, _, c_idx, _, _ = model.backward()

        self._run_check(
            model.W, "W (center)", grad_v_c, c_idx,
            model, centers, contexts, negatives,
        )

    def test_grad_context_positive(self):
        """∂L/∂u_o should match numerical gradient for W_ctx (positive context)."""
        model, centers, contexts, negatives = make_model_and_batch()
        model.forward(centers, contexts, negatives)
        _, grad_u_o, _, _, o_idx, _ = model.backward()

        self._run_check(
            model.W_ctx, "W_ctx (positive)", grad_u_o, o_idx,
            model, centers, contexts, negatives,
        )

    def test_grad_context_negative(self):
        """∂L/∂u_n should match numerical gradient for W_ctx (negative samples)."""
        model, centers, contexts, negatives = make_model_and_batch()
        model.forward(centers, contexts, negatives)
        _, _, grad_u_n, _, _, n_idx = model.backward()

        B, K, D = grad_u_n.shape
        flat_grad = grad_u_n.reshape(-1, D)
        flat_idx = n_idx.reshape(-1)

        self._run_check(
            model.W_ctx, "W_ctx (negative)", flat_grad, flat_idx,
            model, centers, contexts, negatives,
        )

    def test_loss_is_positive(self):
        """SGNS loss should always be non-negative."""
        model, centers, contexts, negatives = make_model_and_batch()
        _, _, loss = model.forward(centers, contexts, negatives)
        assert loss >= 0.0, f"Loss should be non-negative, got {loss}"

    def test_loss_decreases_with_sgd_step(self):
        """A single SGD update should reduce the loss."""
        from src.model.optimizer import SGD

        model, centers, contexts, negatives = make_model_and_batch(seed=7)
        _, _, loss_before = model.forward(centers, contexts, negatives)
        grad_v_c, grad_u_o, grad_u_n, c_idx, o_idx, n_idx = model.backward()

        opt = SGD(lr=0.1)
        opt.step(model, grad_v_c, grad_u_o, grad_u_n, c_idx, o_idx, n_idx)

        _, _, loss_after = model.forward(centers, contexts, negatives)
        assert loss_after < loss_before, (
            f"Loss should decrease after one SGD step: "
            f"before={loss_before:.4f}, after={loss_after:.4f}"
        )

    def test_backward_requires_forward_first(self):
        """backward() without a prior forward() should raise AttributeError."""
        model = Word2Vec(vocab_size=10, embed_dim=4)
        with pytest.raises(AttributeError):
            model.backward()
