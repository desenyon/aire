# Public API freeze (0.3.x)

Until **1.0**, aire may still evolve, but symbols re-exported from `aire` and covered by
`tests/unit/test_api_freeze.py` are treated as a **soft freeze**:

- **Additive** changes (new exports, new optional kwargs with defaults) are fine in minors.
- **Removals / renames** of frozen exports require a **minor** bump + migration note in
  [migration.md](migration.md), and an update to the freeze test.
- **Behaviour tightenings** that raise instead of silent wrong answers are documented in
  [honesty.md](honesty.md) and the CHANGELOG.

Deeper modules (`aire.rag.store`, provider adapters, …) are usable but not all frozen.

See also [public_api.md](public_api.md) and the generated [API reference](api.md).
