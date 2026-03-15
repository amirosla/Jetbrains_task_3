"""Skip-gram pair generation and negative sampling."""

from typing import Iterator

import numpy as np

from .corpus import Vocabulary


class SkipGramDataset:
    """
    Generates (center, context) index pairs for skip-gram training.

    For every position in the token sequence a dynamic context window of
    size in [1, window_size] is sampled (Mikolov et al., 2013).  Smaller
    distances are sampled more often because the window shrinks towards 1
    with probability 1/window_size, giving closer words higher weight.

    Parameters
    ----------
    encoded : np.ndarray of shape (N,)
        Integer token indices (already subsampled).
    window_size : int
        Maximum context window radius on each side of the center word.
    """

    def __init__(self, encoded: np.ndarray, window_size: int = 5) -> None:
        self.encoded = encoded
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.encoded)

    def iterate_pairs(
        self, rng: np.random.Generator
    ) -> Iterator[tuple[int, int]]:
        """
        Yield (center_idx, context_idx) pairs for the full corpus.

        A fresh random window size is drawn for every center word, which
        gives words closer to the center a higher effective sampling weight.
        """
        n = len(self.encoded)
        for i in range(n):
            # Dynamic window: sample window radius uniformly in [1, window_size]
            radius = int(rng.integers(1, self.window_size + 1))
            lo = max(0, i - radius)
            hi = min(n, i + radius + 1)
            center = int(self.encoded[i])
            for j in range(lo, hi):
                if j != i:
                    yield center, int(self.encoded[j])

    def batch_iterator(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (centers, contexts) arrays of shape (batch_size,).

        The last batch may be smaller than ``batch_size``.
        """
        centers, contexts = [], []
        for center, context in self.iterate_pairs(rng):
            centers.append(center)
            contexts.append(context)
            if len(centers) == batch_size:
                yield np.array(centers, dtype=np.int32), np.array(
                    contexts, dtype=np.int32
                )
                centers, contexts = [], []
        if centers:
            yield np.array(centers, dtype=np.int32), np.array(
                contexts, dtype=np.int32
            )


class NegativeSampler:
    """
    Samples negative (noise) word indices according to the unigram^0.75
    distribution (Mikolov et al., 2013, §2.2).

    Raising frequencies to the power 0.75 smooths the distribution towards
    uniform: rare words are sampled more often than pure frequency would
    suggest, preventing the model from being overwhelmed by a handful of
    very frequent words.

    Parameters
    ----------
    vocab : Vocabulary
        Source of word frequency counts.
    power : float
        Exponent applied to unigram frequencies (default 0.75).
    table_size : int
        Size of the internal alias / lookup table.  Larger tables give
        better approximations of the target distribution.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        power: float = 0.75,
        table_size: int = 10_000_000,
    ) -> None:
        self.vocab = vocab
        self.power = power
        self._build_table(table_size)

    def _build_table(self, table_size: int) -> None:
        """Pre-compute a flat lookup table for O(1) negative sampling."""
        counts = self.vocab.word_counts.astype(np.float64) ** self.power
        counts[0] = 0.0  # never sample <UNK> as a negative
        probs = counts / counts.sum()

        # Fill table proportionally to smoothed frequency
        table = np.zeros(table_size, dtype=np.int32)
        idx = 0
        cumulative = 0.0
        for word_idx, p in enumerate(probs):
            cumulative += p
            end = int(cumulative * table_size)
            table[idx:end] = word_idx
            idx = end
        # Fill any remainder (rounding artefacts) with the last valid word
        if idx < table_size:
            table[idx:] = len(probs) - 1

        self._table = table

    def sample(
        self,
        n: int,
        exclude: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Draw ``n`` negative word indices, avoiding indices in ``exclude``.

        Parameters
        ----------
        n : int
            Number of negative samples per (center, context) pair.
        exclude : np.ndarray
            Indices that must not appear in the sample (typically the center
            and context word indices for the current training pair).
        rng : np.random.Generator

        Returns
        -------
        np.ndarray of shape (len(batch), n)  or  (n,) for a scalar pair.
        """
        exclude_set = set(exclude.tolist())
        # Over-sample to handle rejects, then trim
        draws = rng.integers(0, len(self._table), size=n * 3)
        samples = self._table[draws]
        valid = [s for s in samples if s not in exclude_set][:n]

        # Fallback: fill remaining slots with uniform samples (very rare)
        while len(valid) < n:
            idx = int(rng.integers(0, self.vocab.size))
            if idx not in exclude_set and idx != 0:
                valid.append(idx)

        return np.array(valid, dtype=np.int32)

    def sample_batch(
        self,
        centers: np.ndarray,
        contexts: np.ndarray,
        num_negatives: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample negatives for a batch of (center, context) pairs.

        Returns
        -------
        np.ndarray of shape (batch_size, num_negatives)
        """
        batch_size = len(centers)
        negatives = np.zeros((batch_size, num_negatives), dtype=np.int32)
        for i in range(batch_size):
            exclude = np.array([centers[i], contexts[i]], dtype=np.int32)
            negatives[i] = self.sample(num_negatives, exclude, rng)
        return negatives
