# FG²-GDN from scratch

This repo is a from-scratch PyTorch implementation of the FG²-GDN memory update and chunked scan structure. The project focuses on the math of the recurrence, the equivalence between sequential and chunked execution, and the kernel-style intuition behind reorganizing the computation for parallel hardware.

Paper:

- FG²-GDN: Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control — https://arxiv.org/abs/2604.19021

## What this project is about

Standard softmax attention keeps an explicit, ever-growing KV cache and each new query attends over the stored history. In contrast, this linear / delta-style family compresses history into a fixed-size recurrent state.

FG²-GDN adds finer-grained control over that recurrent memory update by separating learned decay/filtering from key-specific write corrections, so the state can be updated more selectively.

## What is implemented

This repo implements the update at three levels:

- One step: update the state once using the filtered old state plus one new write correction.
- One chunk: summarize many one-step updates into a single chunk transform `P` and chunk contribution `C`.
- Many chunks: summarize each chunk independently, then compose the chunk summaries in time order to recover the same final state as sequential execution.

## Core equations

One-step compact update:

```text
S_new = A @ S_old + B
```

Chunk summary:

```text
S_out = P @ S_in + C
```

Chunk composition:

```text
P = P_right @ P_left
C = P_right @ C_left + C_right
```

## What was verified

The current tests check:

- mechanistic one-step update == compact one-step update
- one chunk summary == sequential execution over the same chunk
- `apply_chunk_summary(...)` == sequential execution over the same chunk
- full multi-chunk scan == sequential execution over the whole sequence

## Why this is interesting

This project is not just a paper reimplementation. It is also a small study in how to:

- translate a recurrent memory paper into working code
- factor a sequential recurrence into chunk summaries
- preserve correctness while reorganizing the computation
- build intuition for GPU/kernel-friendly parallel scan structure

That is the main signal I want this repo to carry: strong mathematical understanding, plus growing kernel-engineering intuition.

## Repo layout

```text
capstone/
├── src/
│   ├── fg2_reference.py
│   └── fg2_chunked.py
└── tests/
    └── test_fg2.py
```

## Next polish

Planned next steps for the portfolio version:

- clean benchmark script
- short derivation / explanation note
- one visual for chunk composition
- optional Triton/kernel extension
