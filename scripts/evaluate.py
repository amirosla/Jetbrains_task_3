"""
evaluate.py — evaluate pre-trained Word2Vec embeddings.

Usage
-----
    python scripts/evaluate.py --model results/word2vec.npz \
                                --vocab  results/vocab.npy

Options
-------
--top-k     : number of nearest neighbours to print per query word
--words     : query words for nearest-neighbour lookup
--analogies : run the built-in analogy test suite
--plot      : save a 2-D PCA visualisation to results/
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.word2vec import Word2Vec
from src.training.evaluation import (
    BUILTIN_ANALOGIES,
    evaluate_analogies,
    plot_embeddings,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pre-trained Word2Vec embeddings.")
    p.add_argument("--model", default="results/word2vec.npz",
                   help="Path to saved .npz model file.")
    p.add_argument("--vocab", default="results/vocab.npy",
                   help="Path to saved vocab .npy array.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of nearest neighbours to display.")
    p.add_argument("--words", nargs="*", default=None,
                   help="Words to query nearest neighbours for.")
    p.add_argument("--analogies", action="store_true",
                   help="Run built-in analogy evaluation.")
    p.add_argument("--plot", action="store_true",
                   help="Save 2-D PCA plot of selected words.")
    p.add_argument("--output-dir", default="results",
                   help="Directory for output files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model from {args.model} …")
    model = Word2Vec.load(args.model)

    print(f"Loading vocabulary from {args.vocab} …")
    idx2word: list[str] = np.load(args.vocab, allow_pickle=True).tolist()
    word2idx = {w: i for i, w in enumerate(idx2word)}

    print(f"  Vocabulary size : {len(idx2word):,}")
    print(f"  Embedding dim   : {model.embed_dim}")

    embeddings = model.get_embeddings()

    # Nearest-neighbour queries
    query_words = args.words or [
        "king", "queen", "man", "woman",
        "paris", "france", "london", "computer",
    ]
    print(f"\n--- Top-{args.top_k} nearest neighbours ---")
    for word in query_words:
        if word not in word2idx:
            print(f"  {word!r}: not in vocabulary")
            continue
        neighbours = model.most_similar(word2idx[word], idx2word, top_k=args.top_k)
        print(f"\n  '{word}':")
        for rank, (nbr, score) in enumerate(neighbours, 1):
            print(f"    {rank:2d}. {nbr:<20s} {score:.4f}")

    # Analogy evaluation
    if args.analogies:
        print("\n--- Analogy evaluation (built-in suite, 3CosMul) ---")
        results = evaluate_analogies(embeddings, word2idx, BUILTIN_ANALOGIES)
        print(f"  Accuracy  : {results['accuracy']:.2%}")
        print(f"  Coverage  : {results['coverage']:.2%}")
        print(f"  Evaluated : {results['n_evaluated']} / {len(BUILTIN_ANALOGIES)}")

        # Print per-analogy results
        print("\n  Details:")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        E = embeddings / norms
        for a, b, c, d in BUILTIN_ANALOGIES:
            if any(w not in word2idx for w in [a, b, c, d]):
                status = "SKIP (OOV)"
                predicted = "—"
            else:
                ia, ib, ic, id_ = (word2idx[w] for w in [a, b, c, d])
                cos_b = E @ E[ib]
                cos_a = E @ E[ia]
                cos_c = E @ E[ic]
                scores = (cos_b * cos_c) / (cos_a + 1e-3)
                for excl in [ia, ib, ic]:
                    scores[excl] = -np.inf
                pred_idx = int(np.argmax(scores))
                predicted = idx2word[pred_idx]
                status = "OK" if pred_idx == id_ else "WRONG"
            print(f"    {a}:{b}::{c}:?  →  {predicted:<15s}  (expected: {d})  [{status}]")

    # Visualisation
    if args.plot:
        plot_words = (args.words or query_words) + [
            "germany", "france", "italy", "spain",
            "cat", "dog", "tree", "river",
        ]
        os.makedirs(args.output_dir, exist_ok=True)
        plot_path = os.path.join(args.output_dir, "embeddings_pca.png")
        plot_embeddings(embeddings, plot_words, word2idx, save_path=plot_path)


if __name__ == "__main__":
    main()
