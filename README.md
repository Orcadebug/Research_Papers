# Portfolio Projects

This directory contains clean, standalone projects intended to be understandable without the surrounding learning workspace.

## Current project

- [`fg2-gdn/`](fg2-gdn/) — from-scratch PyTorch implementation of the FG²-GDN recurrent-memory update and chunked scan, with correctness tests.

## Required project shape

```text
portfolio/<slug>/
  README.md       # problem, implementation, result, limitation
  src/            # reference implementation
  tests/          # correctness and edge cases
  experiments/    # controlled experimental scripts/results
  benchmarks/     # fair performance measurements
  report/         # figures and concise technical note
```

Each portfolio project should be independently runnable and ideally its own Git repository.

## Publishing

Run `./scripts/portfolio-sync.sh "your commit message"` from this directory to validate the public-file allowlist, commit, and push to GitHub. The learning scaffolding runs this after a portfolio change has passed its checks during an active session.
