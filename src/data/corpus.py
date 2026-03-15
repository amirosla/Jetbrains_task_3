"""Corpus loading, preprocessing, and vocabulary construction."""

import os
import re
import urllib.request
import zipfile
from collections import Counter
from typing import Optional

import numpy as np


# Tokens occurring fewer times than this are replaced with <UNK>
DEFAULT_MIN_COUNT = 5

# Subsampling threshold (Mikolov et al., 2013)
DEFAULT_SUBSAMPLE_T = 1e-5


def download_text8(dest_dir: str = "data/raw") -> str:
    """
    Download the text8 corpus (~100 MB) from mattmahoney.net.

    text8 is the first 100 MB of cleaned Wikipedia text and is the
    canonical benchmark dataset for word embedding methods.

    Returns the path to the extracted text file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "text8.zip")
    txt_path = os.path.join(dest_dir, "text8")

    if os.path.exists(txt_path):
        print(f"text8 already present at {txt_path}")
        return txt_path

    url = "http://mattmahoney.net/dc/text8.zip"
    print(f"Downloading text8 from {url} …")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    os.remove(zip_path)
    print(f"text8 saved to {txt_path}")
    return txt_path


def load_corpus(path: str, max_tokens: Optional[int] = None) -> list[str]:
    """
    Load a plain-text file and return a flat list of lowercase tokens.

    Parameters
    ----------
    path:
        Path to the text file (text8 or any whitespace-separated corpus).
    max_tokens:
        If given, only the first ``max_tokens`` tokens are returned.  Useful
        for quick smoke-tests without loading the full 17 M-token text8 file.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().lower()

    # Keep only alphabetic characters and whitespace (text8 is already clean,
    # but this makes the loader robust to arbitrary plain-text files).
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    return tokens


class Vocabulary:
    """
    Maps tokens to integer indices and stores frequency information.

    Rare tokens (count < min_count) are mapped to a shared <UNK> token so
    that the embedding matrix stays tractable.

    Attributes
    ----------
    word2idx : dict[str, int]
    idx2word : list[str]
    word_counts : np.ndarray  –  raw occurrence counts, shape (vocab_size,)
    """

    UNK = "<UNK>"

    def __init__(self, min_count: int = DEFAULT_MIN_COUNT) -> None:
        self.min_count = min_count
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.word_counts: np.ndarray = np.array([], dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.idx2word)

    def build(self, tokens: list[str]) -> "Vocabulary":
        """
        Build vocabulary from a flat token list.

        Tokens with count < min_count are collapsed to <UNK>.  <UNK> itself
        is always assigned index 0.
        """
        raw_counts = Counter(tokens)

        # Always reserve index 0 for the unknown token
        self.idx2word = [self.UNK]
        self.word2idx = {self.UNK: 0}
        counts_list = [0]  # placeholder for <UNK> — filled in below

        unk_count = 0
        for word, count in sorted(raw_counts.items(), key=lambda x: -x[1]):
            if count < self.min_count:
                unk_count += count
            else:
                idx = len(self.idx2word)
                self.idx2word.append(word)
                self.word2idx[word] = idx
                counts_list.append(count)

        counts_list[0] = unk_count
        self.word_counts = np.array(counts_list, dtype=np.int64)
        return self

    def encode(self, tokens: list[str]) -> np.ndarray:
        """Convert a token list to an integer-index array."""
        unk_idx = self.word2idx[self.UNK]
        return np.array(
            [self.word2idx.get(t, unk_idx) for t in tokens], dtype=np.int32
        )

    def decode(self, indices: np.ndarray) -> list[str]:
        """Convert an integer-index array to a token list."""
        return [self.idx2word[i] for i in indices]


def subsample(
    encoded: np.ndarray,
    vocab: Vocabulary,
    t: float = DEFAULT_SUBSAMPLE_T,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Stochastic subsampling of frequent tokens (Mikolov et al., 2013, §2.3).

    Each token with frequency f is discarded with probability

        P(discard) = 1 - sqrt(t / f)

    where f = count(w) / total_tokens.  This accelerates training and
    improves representations of rare words by reducing co-occurrence noise
    from very common function words (e.g. "the", "a").

    Parameters
    ----------
    encoded : np.ndarray of shape (N,)
        Token-index sequence produced by ``Vocabulary.encode``.
    vocab : Vocabulary
        Provides per-token counts.
    t : float
        Subsampling threshold (default 1e-5, same as original paper).
    rng : np.random.Generator, optional
        For reproducible experiments.

    Returns
    -------
    np.ndarray
        Filtered index sequence (variable length ≤ N).
    """
    if rng is None:
        rng = np.random.default_rng()

    total = vocab.word_counts.sum()
    freqs = vocab.word_counts[encoded] / total
    # Clip to avoid sqrt of negative for extremely rare tokens
    keep_probs = np.minimum(1.0, np.sqrt(t / np.maximum(freqs, 1e-12)))
    mask = rng.random(len(encoded)) < keep_probs
    return encoded[mask]
