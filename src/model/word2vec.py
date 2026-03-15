"""
Word2Vec: Skip-Gram with Negative Sampling (SGNS) — pure NumPy.

Mathematical background
-----------------------
Skip-gram learns two embedding matrices:

    W  (center embeddings)  : shape (V, D)  — also called "input" embeddings
    W' (context embeddings) : shape (V, D)  — also called "output" embeddings

For a (center c, positive context o) pair and K negative samples {n_1,...,n_K}
drawn from the noise distribution, the training objective is to *maximise*:

    J = log σ(u_o · v_c) + Σ_{k=1}^{K} log σ(−u_{n_k} · v_c)

which is equivalent to *minimising* the negative log-likelihood:

    L = −log σ(u_o · v_c) − Σ_{k=1}^{K} log σ(−u_{n_k} · v_c)

where v_c = W[c] and u_o = W'[o] are D-dimensional vectors and
σ(x) = 1 / (1 + e^{−x}) is the sigmoid function.

Gradient derivation (for a single pair)
-----------------------------------------
Let s_o = u_o · v_c  (positive dot product)
    s_k = u_{n_k} · v_c  (negative dot products)

∂L/∂v_c  = (σ(s_o) − 1) · u_o  +  Σ_k σ(s_k) · u_{n_k}
∂L/∂u_o  = (σ(s_o) − 1) · v_c
∂L/∂u_{n_k} = σ(s_k) · v_c

These are accumulated across the batch and applied via the optimizer.

Note: W and W' are trained jointly but independently.  At inference time
only W (center embeddings) is typically used, although averaging W and W'
sometimes gives slightly better similarity scores.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: clips input to avoid exp overflow."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class Word2Vec:
    """
    Skip-Gram with Negative Sampling implemented in pure NumPy.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (|V|).
    embed_dim : int
        Dimensionality of word embedding vectors (D).
    seed : int, optional
        Random seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        rng = np.random.default_rng(seed)

        # Centre-word embeddings W: initialised uniform in (-0.5/D, 0.5/D)
        # following the original word2vec C implementation.
        scale = 0.5 / embed_dim
        self.W: np.ndarray = rng.uniform(
            -scale, scale, (vocab_size, embed_dim)
        ).astype(np.float32)

        # Context embeddings W': initialised to zero (standard choice that
        # keeps the initial loss well-defined without breaking symmetry
        # between context vectors).
        self.W_ctx: np.ndarray = np.zeros(
            (vocab_size, embed_dim), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        centers: np.ndarray,
        contexts: np.ndarray,
        negatives: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute SGNS loss and cache intermediate values for the backward pass.

        Parameters
        ----------
        centers : np.ndarray, shape (B,)
            Batch of center-word indices.
        contexts : np.ndarray, shape (B,)
            Positive context-word indices.
        negatives : np.ndarray, shape (B, K)
            Negative sample indices.

        Returns
        -------
        pos_sig : np.ndarray, shape (B,)
            σ(u_o · v_c) for each pair (used in backward).
        neg_sig : np.ndarray, shape (B, K)
            σ(u_{n_k} · v_c) for each negative sample (used in backward).
        loss : float
            Mean negative log-likelihood over the batch.
        """
        B, K = negatives.shape

        v_c = self.W[centers]           # (B, D) — centre embeddings
        u_o = self.W_ctx[contexts]      # (B, D) — positive context embeddings
        u_n = self.W_ctx[negatives]     # (B, K, D) — negative context embeds

        # Positive scores: dot product for each pair → (B,)
        pos_scores = np.einsum("bd,bd->b", v_c, u_o)

        # Negative scores: dot product of centre with each negative → (B, K)
        # v_c[:, None, :] broadcasts over K negatives
        neg_scores = np.einsum("bd,bkd->bk", v_c, u_n)

        pos_sig = sigmoid(pos_scores)           # (B,)
        neg_sig = sigmoid(neg_scores)           # (B, K)

        # Loss = −log σ(s_pos) − Σ_k log σ(−s_neg)
        #      = −log σ(s_pos) − Σ_k log(1 − σ(s_neg))
        eps = 1e-7  # numerical guard for log(0)
        loss_pos = -np.log(pos_sig + eps)
        loss_neg = -np.log(1.0 - neg_sig + eps)
        loss = float(np.mean(loss_pos + loss_neg.sum(axis=1)))

        # Cache tensors needed for the backward pass
        self._cache = (centers, contexts, negatives, v_c, u_o, u_n,
                       pos_sig, neg_sig)

        return pos_sig, neg_sig, loss

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
        """
        Compute gradients of the loss w.r.t. W and W_ctx.

        Must be called after ``forward``.

        Returns
        -------
        grad_v_c  : np.ndarray, shape (B, D)   ∂L/∂v_c  (centre embeddings)
        grad_u_o  : np.ndarray, shape (B, D)   ∂L/∂u_o  (positive context)
        grad_u_n  : np.ndarray, shape (B, K, D) ∂L/∂u_n (negative context)
        centers   : np.ndarray, shape (B,)      index array for W update
        contexts  : np.ndarray, shape (B,)      index array for W_ctx update
        negatives : np.ndarray, shape (B, K)    index array for W_ctx update
        """
        centers, contexts, negatives, v_c, u_o, u_n, pos_sig, neg_sig = (
            self._cache
        )
        B, K, D = u_n.shape

        # ∂L/∂u_o = (σ(s_o) − 1) · v_c,  shape (B, D)
        # σ(s_o) − 1 is the "error signal" for the positive pair
        pos_err = (pos_sig - 1.0)[:, None]          # (B, 1)
        grad_u_o = pos_err * v_c                     # (B, D)

        # ∂L/∂u_{n_k} = σ(s_k) · v_c,  shape (B, K, D)
        neg_err = neg_sig[:, :, None]                # (B, K, 1)
        grad_u_n = neg_err * v_c[:, None, :]         # (B, K, D)

        # ∂L/∂v_c accumulates contributions from both positive and negatives:
        #   (σ(s_o) − 1) · u_o  +  Σ_k σ(s_k) · u_{n_k}
        grad_v_c = pos_err * u_o                     # (B, D)
        grad_v_c += np.einsum("bk,bkd->bd", neg_sig, u_n)  # (B, D)

        # Average over batch (consistent with mean loss)
        grad_v_c /= B
        grad_u_o /= B
        grad_u_n /= B

        return grad_v_c, grad_u_o, grad_u_n, centers, contexts, negatives

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embeddings(self, average_ctx: bool = False) -> np.ndarray:
        """
        Return the final word embedding matrix of shape (V, D).

        Parameters
        ----------
        average_ctx : bool
            If True, return the element-wise average of W and W_ctx — this
            sometimes improves nearest-neighbour quality at minor extra cost.
        """
        if average_ctx:
            return (self.W + self.W_ctx) / 2.0
        return self.W

    def most_similar(
        self,
        query_idx: int,
        idx2word: list[str],
        top_k: int = 10,
        average_ctx: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar words to ``query_idx`` by cosine similarity.

        Parameters
        ----------
        query_idx : int
            Index of the query word in the vocabulary.
        idx2word : list[str]
            Vocabulary index-to-word mapping.
        top_k : int
            Number of nearest neighbours to return.
        average_ctx : bool
            Whether to average centre and context embeddings.

        Returns
        -------
        list of (word, cosine_similarity) sorted descending.
        """
        E = self.get_embeddings(average_ctx).astype(np.float64)
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        E_normed = E / norms

        query = E_normed[query_idx]
        sims = E_normed @ query

        # Exclude the query word itself
        sims[query_idx] = -np.inf
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        return [(idx2word[i], float(sims[i])) for i in top_indices]

    def analogy(
        self,
        a: int,
        b: int,
        c: int,
        idx2word: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Solve analogy "a is to b as c is to ?"  (i.e. find d ≈ b − a + c).

        Uses 3CosMul (Levy & Goldberg, 2014) for better accuracy than the
        classic 3CosAdd.

        Parameters
        ----------
        a, b, c : int
            Vocabulary indices for the analogy words.
        idx2word : list[str]
        top_k : int

        Returns
        -------
        list of (word, score) sorted descending.
        """
        E = self.get_embeddings().astype(np.float64)
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        E_normed = E / norms

        cos_b = E_normed @ E_normed[b]
        cos_a = E_normed @ E_normed[a]
        cos_c = E_normed @ E_normed[c]

        eps = 1e-3
        scores = (cos_b * cos_c) / (cos_a + eps)

        # Exclude the three query words
        for idx in [a, b, c]:
            scores[idx] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(idx2word[i], float(scores[i])) for i in top_indices]

    def save(self, path: str) -> None:
        """Save both embedding matrices to a compressed .npz file."""
        np.savez_compressed(path, W=self.W, W_ctx=self.W_ctx)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Word2Vec":
        """Load a previously saved model."""
        data = np.load(path)
        vocab_size, embed_dim = data["W"].shape
        model = cls(vocab_size, embed_dim)
        model.W = data["W"]
        model.W_ctx = data["W_ctx"]
        return model
