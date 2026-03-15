# Word2Vec from Scratch — Pure NumPy Implementation

A complete, from-scratch implementation of **Word2Vec Skip-Gram with Negative Sampling (SGNS)** using only NumPy. No PyTorch, no TensorFlow, no ML frameworks — every gradient is derived by hand and computed explicitly.

---

## Overview

This project implements the core Word2Vec training loop as described in:

> Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS.

The goal is to demonstrate a thorough understanding of the optimization procedure: forward pass, loss computation, gradient derivation, and parameter updates — all from first principles.

---

## Task Summary

> Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant.

**Choices made:**
- **Variant:** Skip-Gram with Negative Sampling (SGNS) — the most widely used and studied variant
- **Dataset:** [text8](http://mattmahoney.net/dc/text8.zip) — the canonical word embedding benchmark (~17M tokens of clean Wikipedia text)
- **Optimizer:** SGD with linear LR decay (default) + Adam (optional)

---

## Mathematical Background

### Objective

For each (center word `c`, context word `o`) pair and `K` negative samples `{n₁, …, nₖ}` drawn from the noise distribution, SGNS minimizes the **negative log-likelihood**:

```
L = -log σ(u_o · v_c) - Σ_{k=1}^{K} log σ(-u_{nk} · v_c)
```

where `v_c = W[c]` (center embedding), `u_o = W'[o]` (context embedding), and `σ` is the sigmoid function.

### Gradients

Let `s_o = u_o · v_c` and `s_k = u_{nk} · v_c`:

```
∂L/∂v_c   = (σ(s_o) - 1) · u_o  +  Σ_k σ(s_k) · u_{nk}
∂L/∂u_o   = (σ(s_o) - 1) · v_c
∂L/∂u_{nk} = σ(s_k) · v_c
```

These gradients are implemented in [`src/model/word2vec.py`](src/model/word2vec.py) and verified against numerical approximations via central differences in [`tests/test_gradients.py`](tests/test_gradients.py).

---

## Stack

| Component     | Choice                              |
|---------------|-------------------------------------|
| Language      | Python 3.10+                        |
| Core math     | NumPy (only)                        |
| Corpus        | text8 (auto-downloaded)             |
| Evaluation    | SciPy (Spearman ρ), Matplotlib      |
| Testing       | pytest + numerical gradient checks  |

---

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── corpus.py        # Text loading, vocabulary, subsampling
│   │   └── dataset.py       # Skip-gram pair generation, negative sampling
│   ├── model/
│   │   ├── word2vec.py      # SGNS forward pass, backward pass, utilities
│   │   └── optimizer.py     # SGD (with LR decay) and Adam
│   └── training/
│       ├── trainer.py       # Training loop orchestration
│       └── evaluation.py    # Word similarity, analogy eval, PCA plot
├── scripts/
│   ├── train.py             # CLI training entry point
│   └── evaluate.py          # CLI evaluation entry point
├── tests/
│   ├── test_gradients.py    # Numerical gradient checks (central differences)
│   ├── test_corpus.py       # Vocabulary and subsampling tests
│   ├── test_dataset.py      # Skip-gram pairs and negative sampler tests
│   └── test_model.py        # Forward/backward shapes, save/load, similarity
├── results/                 # Output directory for embeddings and plots
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/agnieszkamiroslaw/Jetbrains_task_3.git
cd Jetbrains_task_3

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt   # includes pytest
```

---

## Running

### Full training on text8

```bash
python scripts/train.py
```

text8 (~100 MB) is downloaded automatically on the first run.

### Quick smoke test (small slice, fast)

```bash
python scripts/train.py --max-tokens 500000 --epochs 1 --embed-dim 50
```

### All configurable options

```
--corpus        Path to a custom plain-text corpus (default: downloads text8)
--max-tokens    Limit corpus to first N tokens
--embed-dim     Embedding dimensionality          (default: 100)
--window        Maximum context window radius     (default: 5)
--negatives     Negative samples per pair         (default: 5)
--batch-size    Pairs per gradient step           (default: 512)
--epochs        Training epochs                   (default: 3)
--lr            Initial learning rate             (default: 0.025)
--min-lr        Final learning rate (SGD decay)   (default: 1e-4)
--min-count     Minimum token frequency           (default: 5)
--subsample-t   Subsampling threshold             (default: 1e-5)
--optimizer     sgd | adam                        (default: sgd)
--seed          Random seed                       (default: 42)
--output-dir    Save directory                    (default: results/)
--eval-words    Words to show neighbours for
--plot          Save 2-D PCA visualisation
```

### Evaluate saved embeddings

```bash
python scripts/evaluate.py --model results/word2vec.npz \
                            --vocab  results/vocab.npy \
                            --analogies --plot
```

---

## Testing

```bash
pytest tests/ -v
```

Expected output: **40 tests passed**.

### What is tested

| File | What |
|------|------|
| `test_gradients.py` | Analytical vs. numerical (central-difference) gradient checks for W, W_ctx (positive and negative). Also verifies loss decreases after one SGD step. |
| `test_corpus.py` | Vocabulary construction, `<UNK>` handling, subsampling frequency properties. |
| `test_dataset.py` | Skip-gram pair window bounds, no self-pairs, batch shapes, pair completeness, negative sampler constraints, distribution skew. |
| `test_model.py` | Sigmoid edge cases, embedding matrix shapes, forward/backward output types and shapes, save/load roundtrip, most-similar ordering. |

---

## Assumptions

1. **SGNS over CBOW** — Skip-gram with negative sampling is chosen because it produces higher-quality embeddings for rare words and scales better than hierarchical softmax, making it the standard choice in practice.

2. **Two embedding matrices** — Following the original implementation, separate center (`W`) and context (`W'`) matrices are maintained. Only `W` is returned by default at inference; averaging `W` and `W'` is supported via `get_embeddings(average_ctx=True)`.

3. **text8 as default corpus** — text8 is the established community benchmark for word embedding implementations, enabling direct comparison with published results without requiring licensing agreements.

4. **min\_count = 5** — Tokens occurring fewer than 5 times are treated as `<UNK>`. This is the standard threshold used in the original word2vec codebase.

5. **Dynamic window** — A random window radius in `[1, window_size]` is sampled per center word, giving closer words higher effective weight. This matches the original implementation.

6. **W\_ctx initialized to zero** — Center embeddings W are initialized uniform in `(-0.5/D, 0.5/D)` (matching the original C code); context embeddings W_ctx start at zero, which is a stable initialization that avoids symmetry-breaking issues with two random matrices.

---

## Design Decisions

### Batched forward/backward
The forward and backward passes operate on mini-batches using `np.einsum`, avoiding Python loops over individual pairs and achieving acceptable throughput on CPU.

### Sparse optimizer updates
Only rows of `W` and `W_ctx` that appear in the current batch are updated. This is essential for efficiency — full-matrix updates would be `O(V × D)` per step regardless of batch size.

### Unigram^0.75 table
Negative sampling uses a pre-computed lookup table of size 10M filled proportionally to `count^0.75`. This gives O(1) sampling while accurately approximating the smoothed unigram distribution.

### Numerical gradient verification
The test suite includes central-difference gradient checks (`ε = 1e-4`) for all three parameter groups (center, positive context, negative context). This provides strong evidence of correctness without relying on autograd.

### No external ML dependencies
The only numerics dependency is NumPy. SciPy is used only for `spearmanr` in evaluation (not training). This makes the implementation fully transparent and portable.

---

## Trade-offs

| Decision | Advantage | Cost |
|----------|-----------|------|
| Pure NumPy (no autograd) | Full transparency, no hidden ops | Slower than GPU-backed frameworks; gradients must be derived manually |
| Batched NumPy einsum | ~10× faster than Python loops | Memory grows with batch size |
| Sparse Adam | Correct moment estimates for embeddings | More complex indexing than dense Adam |
| Lookup-table negative sampling | O(1) sampling | ~40 MB of RAM for table; slight approximation vs. true multinomial |
| Two matrices (W, W_ctx) | Standard and well-studied | Using only W loses some signal; averaging is optional |

---

## Future Improvements

- **Subword embeddings (fastText)** — extend the model to handle out-of-vocabulary words via character n-gram hashing
- **Hierarchical Softmax** — alternative to negative sampling that avoids the noise distribution hyperparameter
- **CBOW variant** — averaging context vectors to predict the center word; simpler forward pass, slightly lower quality on rare words
- **Multi-threaded data loading** — Python's GIL limits pure-NumPy parallelism; a C extension or multiprocessing queue would significantly improve throughput
- **Streaming corpus reader** — avoid loading the full corpus into RAM; enables training on arbitrarily large text files
- **WordSim-353 / SimLex-999 evaluation** — integrate standard benchmark datasets for quantitative similarity evaluation
- **Google analogy dataset** — full 19,544-pair analogy suite for rigorous evaluation
- **Learning rate warmup** — especially useful with Adam to stabilize early training
