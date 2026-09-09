# Contributing to flybrain-sdk

Issues, ideas, model-source additions and pull requests are welcome. You do not
need a CUDA device. Small, focused changes are easiest to review.

1. Fork the public repository and create a branch in your fork.
2. Install with `python -m pip install -e ".[dev]"`.
3. Make a focused change and add relevant tests/examples/documentation.
4. Run `pytest`, `ruff check .`, `ruff format --check .`, and the quickstart.
5. Open a pull request explaining the change, evidence and limitations.

For TypeScript contract changes, run `npm ci && npm run typecheck` in `packages/js`.
Python CI must stay network-independent: mock download responses, never fetch a
full dataset in tests. For a deliberate live smoke test, use a small catalog asset
and a temporary cache. Model-source submissions need version, URL, size, license,
source citations, integrity metadata and an honest raw/runnable status.

Changes to numerical semantics require tests of observed behavior, an explanation
of compatibility and, when necessary, a new dynamics revision. Add benchmarks
before making performance claims. Avoid large biological datasets in Git.

Maintainers review contributions before merging. Submitting a PR does not grant
write access to the main branch. No CLA is required; contributions are provided
under the repository's MIT license unless explicitly identified as separately
licensed data. You must have the right to contribute the material.

Please be respectful: discuss ideas and evidence, avoid harassment and private
information, and make space for beginners. The maintainer may remove abusive
content or close off-topic submissions. Use GitHub's private vulnerability reporting
when enabled; otherwise report security-sensitive details privately to the owner
through an available verified contact, rather than in a public issue.
