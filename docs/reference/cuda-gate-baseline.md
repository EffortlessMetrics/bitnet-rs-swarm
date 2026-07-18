# CUDA feature-gate baseline

Reconciles the `crates/bitnet-kernels/tests/feature_gate_consistency.rs`
consistency checks (originally issue #439) with the current tree, per
issue #1709.

## Policy

New GPU code should use the unified predicate

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```

rather than a standalone `#[cfg(feature = "cuda")]`, so that a non-CUDA `gpu`
backend and the CUDA backend stay in sync at compile time.

However, the tree already contains many standalone `#[cfg(feature = "cuda")]`
gates. Most live under `crates/bitnet-kernels/src/cuda/**` (and other
CUDA-specific surfaces such as `bitnet-device-probe/src/nvidia_cuda.rs`) where a
bare `feature = "cuda"` gate is **correct**: unifying it to
`any(feature = "gpu", feature = "cuda")` would pull CUDA-only code into a
non-CUDA `gpu` build and fail to compile. Rewriting those is neither safe nor
desirable.

So the check is a **grandfathered baseline** rather than a hard ban:

- Pre-existing findings are recorded in
  `crates/bitnet-kernels/tests/cuda_gate_baseline.tsv`.
- The tests fail only on **new** standalone gates beyond the recorded
  per-identity counts (no-new-debt).
- Reducing the count (fixing debt) is always allowed.

Identities are `path` + the matched construct only (ripgrep runs with
`--only-matching`, so the identity excludes line numbers, indentation, and any
trailing comment). This keeps the check stable across unrelated edits and
prevents an incidental `any(feature ...)` in a comment from masking a real gate.
Each occurrence on a shared line is counted separately.

## Exceptions

Two categories bypass the baseline entirely:

- Runtime `cfg!(feature = "cuda")` checks in the authoritative sources listed in
  `ALLOWED_CFG_MACRO_EXCEPTIONS` (`kernel_registry.rs`, `backend_selection.rs`,
  `bitnet-runtime-feature-flags/src/lib.rs`), which deliberately distinguish
  CUDA from the `gpu` umbrella.
- Gates already written with the unified `any(feature = ...)` predicate.

## Regenerating the baseline

After an intentional, reviewed change to CUDA-specific gates:

```bash
BLESS_CUDA_GATE_BASELINE=1 \
  cargo test --locked -p bitnet-kernels --no-default-features \
  --test feature_gate_consistency -- --test-threads=1
```

Review the resulting `cuda_gate_baseline.tsv` diff before committing — it should
only add entries you intend to grandfather.

## Validation

```bash
cargo test --locked -p bitnet-kernels --no-default-features \
  --test feature_gate_consistency
```

The `baseline_matcher` unit tests cover the no-new-debt matcher itself
(new identity flagged, extra occurrence flagged, debt removal allowed).
