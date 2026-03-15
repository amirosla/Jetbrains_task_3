"""
Intrinsic evaluation of word embeddings.

Two standard evaluation tasks are implemented:

1. **Word similarity** — Spearman ρ between model cosine similarities and
   human-annotated word-pair scores (e.g. WordSim-353, SimLex-999).

2. **Word analogy** — 3CosMul accuracy on the Google analogy dataset
   (Mikolov et al., 2013).

Neither task requires external network access; a small built-in analogy
suite is shipped with the module for quick smoke-testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Cosine similarity utility
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Word-similarity benchmark
# ---------------------------------------------------------------------------

def evaluate_similarity(
    embeddings: np.ndarray,
    word2idx: dict[str, int],
    pairs: list[tuple[str, str, float]],
) -> dict[str, float]:
    """
    Compute Spearman ρ between model and human word-pair scores.

    Parameters
    ----------
    embeddings : np.ndarray, shape (V, D)
    word2idx : dict[str, int]
    pairs : list of (word1, word2, human_score)

    Returns
    -------
    dict with keys:
        ``spearman_rho``   — Spearman correlation coefficient
        ``coverage``       — fraction of pairs where both words are in vocab
        ``n_evaluated``    — number of pairs scored
    """
    model_scores, human_scores = [], []

    for w1, w2, human_score in pairs:
        if w1 not in word2idx or w2 not in word2idx:
            continue
        e1 = embeddings[word2idx[w1]]
        e2 = embeddings[word2idx[w2]]
        model_scores.append(cosine_similarity(e1, e2))
        human_scores.append(human_score)

    if len(model_scores) < 2:
        return {"spearman_rho": float("nan"), "coverage": 0.0, "n_evaluated": 0}

    rho, _ = spearmanr(model_scores, human_scores)
    coverage = len(model_scores) / len(pairs)

    return {
        "spearman_rho": float(rho),
        "coverage": float(coverage),
        "n_evaluated": len(model_scores),
    }


# ---------------------------------------------------------------------------
# Word-analogy benchmark (3CosMul)
# ---------------------------------------------------------------------------

def evaluate_analogies(
    embeddings: np.ndarray,
    word2idx: dict[str, int],
    analogies: list[tuple[str, str, str, str]],
) -> dict[str, float]:
    """
    Evaluate word analogies using 3CosMul (Levy & Goldberg, 2014).

    Each analogy is a tuple (a, b, c, d) representing "a:b :: c:d".
    The model must predict d given a, b, c.

    Parameters
    ----------
    embeddings : np.ndarray, shape (V, D)
    word2idx : dict[str, int]
    analogies : list of (a, b, c, d) string tuples

    Returns
    -------
    dict with keys:
        ``accuracy``     — fraction of correctly predicted analogies
        ``coverage``     — fraction of analogies where all 4 words are in vocab
        ``n_evaluated``  — number of analogies scored
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    E = embeddings / norms   # (V, D) normalised

    correct = 0
    total = 0

    for a, b, c, d in analogies:
        if any(w not in word2idx for w in [a, b, c, d]):
            continue

        ia, ib, ic, id_ = (word2idx[w] for w in [a, b, c, d])

        # 3CosMul: argmax  cos(x, b) * cos(x, c) / (cos(x, a) + ε)
        cos_b = E @ E[ib]
        cos_a = E @ E[ia]
        cos_c = E @ E[ic]

        scores = (cos_b * cos_c) / (cos_a + 1e-3)
        # Exclude query words from candidates
        for excl in [ia, ib, ic]:
            scores[excl] = -np.inf

        predicted = int(np.argmax(scores))
        if predicted == id_:
            correct += 1
        total += 1

    if total == 0:
        return {"accuracy": float("nan"), "coverage": 0.0, "n_evaluated": 0}

    coverage = total / len(analogies)
    return {
        "accuracy": correct / total,
        "coverage": float(coverage),
        "n_evaluated": total,
    }


# ---------------------------------------------------------------------------
# Built-in mini analogy suite (for quick smoke-testing without downloads)
# ---------------------------------------------------------------------------

BUILTIN_ANALOGIES: list[tuple[str, str, str, str]] = [
    # Capitals
    ("berlin", "germany", "paris", "france"),
    ("berlin", "germany", "rome", "italy"),
    ("berlin", "germany", "madrid", "spain"),
    ("berlin", "germany", "athens", "greece"),
    # Plural / singular
    ("man", "men", "woman", "women"),
    ("child", "children", "foot", "feet"),
    # Comparative / superlative
    ("good", "better", "bad", "worse"),
    ("big", "bigger", "small", "smaller"),
    # Currency
    ("germany", "euro", "japan", "yen"),
    ("britain", "pound", "japan", "yen"),
]


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def plot_embeddings(
    embeddings: np.ndarray,
    words: list[str],
    word2idx: dict[str, int],
    save_path: Optional[str] = None,
) -> None:
    """
    Project a subset of embeddings to 2-D with PCA and plot them.

    Parameters
    ----------
    embeddings : np.ndarray, shape (V, D)
    words : list[str]
        Words to include in the plot.
    word2idx : dict[str, int]
    save_path : str, optional
        If given, save the figure here instead of displaying it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    valid = [(w, word2idx[w]) for w in words if w in word2idx]
    if not valid:
        print("None of the requested words are in the vocabulary.")
        return

    labels, indices = zip(*valid)
    vecs = embeddings[list(indices)].astype(np.float64)

    # PCA to 2-D (manual implementation — no sklearn dependency)
    vecs -= vecs.mean(axis=0)
    cov = vecs.T @ vecs / len(vecs)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take the two largest eigenvectors
    top2 = eigenvectors[:, np.argsort(-eigenvalues)[:2]]
    projected = vecs @ top2

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(projected[:, 0], projected[:, 1], alpha=0.7, s=40)
    for i, label in enumerate(labels):
        ax.annotate(label, (projected[i, 0], projected[i, 1]),
                    fontsize=9, alpha=0.9)
    ax.set_title("Word Embeddings (PCA 2D Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
