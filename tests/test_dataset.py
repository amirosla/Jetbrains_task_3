"""Unit tests for SkipGramDataset and NegativeSampler."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.corpus import Vocabulary
from src.data.dataset import NegativeSampler, SkipGramDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_vocab():
    tokens = ["a"] * 50 + ["b"] * 40 + ["c"] * 30 + ["d"] * 20 + ["e"] * 10
    return Vocabulary(min_count=1).build(tokens)


@pytest.fixture
def tiny_encoded():
    # Deterministic sequence of indices 0-9
    return np.arange(10, dtype=np.int32)


# ---------------------------------------------------------------------------
# SkipGramDataset
# ---------------------------------------------------------------------------

class TestSkipGramDataset:
    def test_pairs_within_window(self, tiny_encoded):
        """All generated pairs must be within ``window_size`` distance."""
        window = 2
        dataset = SkipGramDataset(tiny_encoded, window_size=window)
        rng = np.random.default_rng(0)
        for center, context in dataset.iterate_pairs(rng):
            # Recover positions
            center_pos = int(np.where(tiny_encoded == center)[0][0])
            context_pos = int(np.where(tiny_encoded == context)[0][0])
            assert abs(center_pos - context_pos) <= window, (
                f"Pair ({center}, {context}) exceeds window {window}"
            )

    def test_no_self_pairs(self, tiny_encoded):
        """A word should never be its own context."""
        dataset = SkipGramDataset(tiny_encoded, window_size=3)
        rng = np.random.default_rng(0)
        for center, context in dataset.iterate_pairs(rng):
            assert center != context or len(np.unique(tiny_encoded)) == 1

    def test_batch_shapes(self, tiny_encoded):
        dataset = SkipGramDataset(tiny_encoded, window_size=2)
        rng = np.random.default_rng(0)
        for centers, contexts in dataset.batch_iterator(batch_size=4, rng=rng):
            assert centers.shape == contexts.shape
            assert len(centers) <= 4

    def test_all_pairs_emitted(self, tiny_encoded):
        """batch_iterator should emit the same pairs as iterate_pairs."""
        dataset = SkipGramDataset(tiny_encoded, window_size=2)

        rng1 = np.random.default_rng(99)
        all_pairs_iter = list(dataset.iterate_pairs(rng1))

        rng2 = np.random.default_rng(99)
        all_pairs_batch = []
        for c, ctx in dataset.batch_iterator(batch_size=16, rng=rng2):
            for i in range(len(c)):
                all_pairs_batch.append((int(c[i]), int(ctx[i])))

        assert all_pairs_iter == all_pairs_batch


# ---------------------------------------------------------------------------
# NegativeSampler
# ---------------------------------------------------------------------------

class TestNegativeSampler:
    def test_sample_count(self, tiny_vocab):
        sampler = NegativeSampler(tiny_vocab)
        rng = np.random.default_rng(0)
        exclude = np.array([1, 2], dtype=np.int32)
        samples = sampler.sample(n=5, exclude=exclude, rng=rng)
        assert len(samples) == 5

    def test_no_unk_in_samples(self, tiny_vocab):
        """<UNK> (index 0) must never be returned as a negative sample."""
        sampler = NegativeSampler(tiny_vocab)
        rng = np.random.default_rng(0)
        exclude = np.array([1], dtype=np.int32)
        for _ in range(50):
            samples = sampler.sample(n=10, exclude=exclude, rng=rng)
            assert 0 not in samples, "NegativeSampler returned <UNK>"

    def test_excluded_words_absent(self, tiny_vocab):
        """Excluded words must not appear in negative samples."""
        sampler = NegativeSampler(tiny_vocab)
        rng = np.random.default_rng(0)
        exclude = np.array([1, 2, 3], dtype=np.int32)
        for _ in range(30):
            samples = sampler.sample(n=5, exclude=exclude, rng=rng)
            for e in exclude:
                assert e not in samples, f"Excluded index {e} found in samples"

    def test_batch_sample_shape(self, tiny_vocab):
        sampler = NegativeSampler(tiny_vocab)
        rng = np.random.default_rng(0)
        B, K = 8, 5
        centers = np.zeros(B, dtype=np.int32)
        contexts = np.ones(B, dtype=np.int32)
        negatives = sampler.sample_batch(centers, contexts, K, rng)
        assert negatives.shape == (B, K)

    def test_distribution_roughly_correct(self, tiny_vocab):
        """
        Sampling distribution should be skewed towards higher-frequency words
        (with the 0.75 smoothing).  The most frequent non-UNK word should be
        sampled at least as often as the least frequent word.
        """
        sampler = NegativeSampler(tiny_vocab, table_size=100_000)
        rng = np.random.default_rng(42)
        samples = sampler.sample(n=10_000, exclude=np.array([0]), rng=rng)
        counts = np.bincount(samples, minlength=tiny_vocab.size)
        # Most frequent word (rank 1 by count) should be sampled more often
        sorted_by_freq = np.argsort(-tiny_vocab.word_counts[1:]) + 1  # skip UNK
        most_freq_idx = sorted_by_freq[0]
        least_freq_idx = sorted_by_freq[-1]
        assert counts[most_freq_idx] >= counts[least_freq_idx], (
            "Most frequent word should be sampled at least as often as least frequent"
        )
