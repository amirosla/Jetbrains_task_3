"""
train.py — entry point for training Word2Vec on text8 (or any plain-text corpus).

Usage examples
--------------
# Full text8 training with defaults:
    python scripts/train.py

# Quick smoke test on a small slice:
    python scripts/train.py --max-tokens 200000 --epochs 1 --embed-dim 50

# Use Adam instead of SGD:
    python scripts/train.py --optimizer adam --lr 1e-3

# Specify a custom corpus file:
    python scripts/train.py --corpus path/to/my_corpus.txt
"""

import argparse
import os
import sys

import numpy as np

# Ensure the project root is on the path when called as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.corpus import Vocabulary, download_text8, load_corpus, subsample
from src.data.dataset import NegativeSampler, SkipGramDataset
from src.model.optimizer import SGD, Adam
from src.model.word2vec import Word2Vec
from src.training.evaluation import (
    BUILTIN_ANALOGIES,
    evaluate_analogies,
    plot_embeddings,
)
from src.training.trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Word2Vec (skip-gram + negative sampling) on text8."
    )
    p.add_argument("--corpus", type=str, default=None,
                   help="Path to a plain-text corpus. Downloads text8 if omitted.")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Limit corpus to first N tokens (useful for quick tests).")
    p.add_argument("--embed-dim", type=int, default=100,
                   help="Embedding dimensionality (default: 100).")
    p.add_argument("--window", type=int, default=5,
                   help="Maximum context window size (default: 5).")
    p.add_argument("--negatives", type=int, default=5,
                   help="Number of negative samples per pair (default: 5).")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size (default: 512).")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs (default: 3).")
    p.add_argument("--lr", type=float, default=0.025,
                   help="Initial learning rate (default: 0.025).")
    p.add_argument("--min-lr", type=float, default=1e-4,
                   help="Final learning rate for SGD decay (default: 1e-4).")
    p.add_argument("--min-count", type=int, default=5,
                   help="Minimum token frequency to include in vocab (default: 5).")
    p.add_argument("--subsample-t", type=float, default=1e-5,
                   help="Subsampling threshold (default: 1e-5).")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                   help="Optimizer to use (default: sgd).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for saving embeddings and plots.")
    p.add_argument("--eval-words", nargs="*", default=None,
                   help="Words to show nearest neighbours for after training.")
    p.add_argument("--plot", action="store_true",
                   help="Save a 2-D PCA visualisation of embeddings.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    if args.corpus:
        corpus_path = args.corpus
    else:
        corpus_path = download_text8()

    print(f"\n[1/5] Loading corpus from {corpus_path} …")
    tokens = load_corpus(corpus_path, max_tokens=args.max_tokens)
    print(f"      Total tokens loaded: {len(tokens):,}")

    # ------------------------------------------------------------------
    # 2. Build vocabulary
    # ------------------------------------------------------------------
    print(f"\n[2/5] Building vocabulary (min_count={args.min_count}) …")
    vocab = Vocabulary(min_count=args.min_count).build(tokens)
    print(f"      Vocabulary size: {vocab.size:,}")

    # ------------------------------------------------------------------
    # 3. Encode + subsample
    # ------------------------------------------------------------------
    print(f"\n[3/5] Encoding and subsampling (t={args.subsample_t}) …")
    encoded = vocab.encode(tokens)
    encoded = subsample(encoded, vocab, t=args.subsample_t, rng=rng)
    print(f"      Tokens after subsampling: {len(encoded):,}  "
          f"(kept {100 * len(encoded) / len(tokens):.1f}%)")

    dataset = SkipGramDataset(encoded, window_size=args.window)
    sampler = NegativeSampler(vocab)

    # ------------------------------------------------------------------
    # 4. Initialise model & optimizer
    # ------------------------------------------------------------------
    print(f"\n[4/5] Initialising model  "
          f"(V={vocab.size:,}, D={args.embed_dim}) …")
    model = Word2Vec(vocab.size, args.embed_dim, seed=args.seed)

    # Rough estimate of total batches for LR decay schedule
    pairs_per_epoch = len(encoded) * args.window  # upper bound
    total_steps = (pairs_per_epoch // args.batch_size + 1) * args.epochs

    if args.optimizer == "sgd":
        optimizer = SGD(lr=args.lr, min_lr=args.min_lr, total_steps=total_steps)
    else:
        optimizer = Adam(lr=args.lr)

    config = TrainingConfig(
        embed_dim=args.embed_dim,
        window_size=args.window,
        num_negatives=args.negatives,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        seed=args.seed,
        checkpoint_dir=None,
    )

    trainer = Trainer(model, dataset, sampler, optimizer, config)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"\n[5/5] Training for {args.epochs} epoch(s) …\n")
    loss_history = trainer.train()

    if loss_history:
        print(f"\nFinal mean loss: {loss_history[-1]:.4f}")

    # ------------------------------------------------------------------
    # Save embeddings
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "word2vec.npz")
    model.save(model_path)

    vocab_path = os.path.join(args.output_dir, "vocab.npy")
    np.save(vocab_path, np.array(vocab.idx2word))
    print(f"Vocabulary saved to {vocab_path}")

    # Save loss curve
    if loss_history:
        loss_path = os.path.join(args.output_dir, "loss_history.npy")
        np.save(loss_path, np.array(loss_history))
        print(f"Loss history saved to {loss_path}")

    # ------------------------------------------------------------------
    # Quick evaluation
    # ------------------------------------------------------------------
    embeddings = model.get_embeddings()
    word2idx = vocab.word2idx

    eval_words = args.eval_words or ["king", "queen", "man", "woman",
                                      "paris", "london", "python"]
    print("\n--- Nearest neighbours ---")
    for word in eval_words:
        if word not in word2idx:
            print(f"  {word!r}: not in vocabulary")
            continue
        neighbours = model.most_similar(word2idx[word], vocab.idx2word, top_k=5)
        nbr_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
        print(f"  {word}: {nbr_str}")

    print("\n--- Analogy evaluation (3CosMul) ---")
    analogy_results = evaluate_analogies(embeddings, word2idx, BUILTIN_ANALOGIES)
    print(f"  Accuracy  : {analogy_results['accuracy']:.2%}")
    print(f"  Coverage  : {analogy_results['coverage']:.2%}")
    print(f"  Evaluated : {analogy_results['n_evaluated']} / {len(BUILTIN_ANALOGIES)}")

    if args.plot:
        plot_words = eval_words + ["computer", "science", "the", "a",
                                    "germany", "france", "italy"]
        plot_path = os.path.join(args.output_dir, "embeddings_pca.png")
        plot_embeddings(embeddings, plot_words, word2idx, save_path=plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
