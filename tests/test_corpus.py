"""Unit tests for corpus loading and vocabulary construction."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.corpus import Vocabulary, subsample


class TestVocabulary:
    """Tests for the Vocabulary class."""

    @pytest.fixture
    def small_tokens(self):
        return (
            ["the"] * 100
            + ["cat"] * 50
            + ["sat"] * 30
            + ["on"] * 20
            + ["mat"] * 10
            + ["rare"] * 2   # below min_count=5 → goes to <UNK>
        )

    def test_unk_is_index_zero(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        assert vocab.word2idx[Vocabulary.UNK] == 0

    def test_rare_words_excluded(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        assert "rare" not in vocab.word2idx

    def test_vocab_size(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        # Expect: <UNK>, the, cat, sat, on, mat  → size 6
        assert vocab.size == 6

    def test_encode_decode_roundtrip(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        encoded = vocab.encode(["the", "cat"])
        decoded = vocab.decode(encoded)
        assert decoded == ["the", "cat"]

    def test_unknown_token_maps_to_unk(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        encoded = vocab.encode(["unknownword"])
        assert int(encoded[0]) == vocab.word2idx[Vocabulary.UNK]

    def test_word_counts_shape(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        assert vocab.word_counts.shape == (vocab.size,)

    def test_most_frequent_word_has_correct_count(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        idx = vocab.word2idx["the"]
        assert int(vocab.word_counts[idx]) == 100

    def test_unk_absorbs_rare_counts(self, small_tokens):
        vocab = Vocabulary(min_count=5).build(small_tokens)
        # "rare" appears 2 times → should be in <UNK> count
        assert vocab.word_counts[0] >= 2


class TestSubsampling:
    """Tests for the subsampling procedure."""

    def test_output_is_shorter_or_equal(self):
        tokens = ["the"] * 1000 + ["cat"] * 50
        vocab = Vocabulary(min_count=5).build(tokens)
        encoded = vocab.encode(tokens)
        rng = np.random.default_rng(0)
        subsampled = subsample(encoded, vocab, t=1e-5, rng=rng)
        assert len(subsampled) <= len(encoded)

    def test_high_threshold_keeps_all_tokens(self):
        """With t=1 almost all tokens should be retained."""
        tokens = ["word_a"] * 100 + ["word_b"] * 100
        vocab = Vocabulary(min_count=5).build(tokens)
        encoded = vocab.encode(tokens)
        rng = np.random.default_rng(0)
        # t=1 makes keep_prob ≈ sqrt(1/f) ≥ 1 for f≤1 → all kept
        subsampled = subsample(encoded, vocab, t=1.0, rng=rng)
        assert len(subsampled) == len(encoded)

    def test_very_frequent_words_are_downsampled(self):
        """
        Words making up >50% of the corpus should be aggressively dropped
        with the default threshold.
        """
        tokens = ["the"] * 10_000 + ["python"] * 100
        vocab = Vocabulary(min_count=5).build(tokens)
        encoded = vocab.encode(tokens)
        rng = np.random.default_rng(42)
        subsampled = subsample(encoded, vocab, t=1e-5, rng=rng)
        the_idx = vocab.word2idx["the"]
        original_the_count = int((encoded == the_idx).sum())
        kept_the_count = int((subsampled == the_idx).sum())
        assert kept_the_count < original_the_count * 0.5
