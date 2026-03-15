"""Unit tests for Word2Vec model: forward pass, shapes, save/load."""

import os
import tempfile

import numpy as np
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.word2vec import Word2Vec, sigmoid


class TestSigmoid:
    def test_sigmoid_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_no_nan_for_extreme_inputs(self):
        x = np.array([-1000.0, 0.0, 1000.0])
        result = sigmoid(x)
        assert not np.any(np.isnan(result))


class TestWord2VecShapes:
    @pytest.fixture
    def model(self):
        return Word2Vec(vocab_size=50, embed_dim=16, seed=0)

    def test_W_shape(self, model):
        assert model.W.shape == (50, 16)

    def test_W_ctx_shape(self, model):
        assert model.W_ctx.shape == (50, 16)

    def test_W_ctx_initialised_to_zero(self, model):
        assert np.allclose(model.W_ctx, 0.0)

    def test_forward_returns_correct_shapes(self, model):
        rng = np.random.default_rng(0)
        B, K = 8, 5
        centers = rng.integers(0, 50, B).astype(np.int32)
        contexts = rng.integers(0, 50, B).astype(np.int32)
        negatives = rng.integers(0, 50, (B, K)).astype(np.int32)

        pos_sig, neg_sig, loss = model.forward(centers, contexts, negatives)
        assert pos_sig.shape == (B,)
        assert neg_sig.shape == (B, K)
        assert isinstance(loss, float)

    def test_pos_sig_values_in_01(self, model):
        """Sigmoid outputs must be in (0, 1)."""
        rng = np.random.default_rng(1)
        centers = rng.integers(0, 50, 16).astype(np.int32)
        contexts = rng.integers(0, 50, 16).astype(np.int32)
        negatives = rng.integers(0, 50, (16, 5)).astype(np.int32)
        pos_sig, neg_sig, _ = model.forward(centers, contexts, negatives)
        assert np.all(pos_sig > 0) and np.all(pos_sig < 1)
        assert np.all(neg_sig > 0) and np.all(neg_sig < 1)

    def test_backward_gradient_shapes(self, model):
        rng = np.random.default_rng(2)
        B, K = 8, 5
        centers = rng.integers(0, 50, B).astype(np.int32)
        contexts = rng.integers(0, 50, B).astype(np.int32)
        negatives = rng.integers(0, 50, (B, K)).astype(np.int32)

        model.forward(centers, contexts, negatives)
        grad_v_c, grad_u_o, grad_u_n, c_idx, o_idx, n_idx = model.backward()

        assert grad_v_c.shape == (B, 16)
        assert grad_u_o.shape == (B, 16)
        assert grad_u_n.shape == (B, K, 16)
        assert c_idx.shape == (B,)
        assert o_idx.shape == (B,)
        assert n_idx.shape == (B, K)


class TestWord2VecSaveLoad:
    def test_save_and_load(self):
        model = Word2Vec(vocab_size=30, embed_dim=8, seed=5)
        rng = np.random.default_rng(5)
        model.W = rng.standard_normal((30, 8)).astype(np.float32)
        model.W_ctx = rng.standard_normal((30, 8)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model")
            model.save(path)
            loaded = Word2Vec.load(path + ".npz")

        assert np.allclose(model.W, loaded.W)
        assert np.allclose(model.W_ctx, loaded.W_ctx)
        assert loaded.vocab_size == model.vocab_size
        assert loaded.embed_dim == model.embed_dim


class TestMostSimilar:
    def test_most_similar_returns_top_k(self):
        model = Word2Vec(vocab_size=20, embed_dim=8, seed=3)
        rng = np.random.default_rng(3)
        model.W = rng.standard_normal((20, 8)).astype(np.float32)
        idx2word = [f"word_{i}" for i in range(20)]
        results = model.most_similar(0, idx2word, top_k=5)
        assert len(results) == 5

    def test_query_word_not_in_results(self):
        model = Word2Vec(vocab_size=20, embed_dim=8, seed=3)
        rng = np.random.default_rng(3)
        model.W = rng.standard_normal((20, 8)).astype(np.float32)
        idx2word = [f"word_{i}" for i in range(20)]
        results = model.most_similar(0, idx2word, top_k=10)
        result_words = [w for w, _ in results]
        assert "word_0" not in result_words, "Query word should not appear in results"

    def test_similarities_are_descending(self):
        model = Word2Vec(vocab_size=20, embed_dim=8, seed=3)
        rng = np.random.default_rng(3)
        model.W = rng.standard_normal((20, 8)).astype(np.float32)
        idx2word = [f"word_{i}" for i in range(20)]
        results = model.most_similar(0, idx2word, top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
