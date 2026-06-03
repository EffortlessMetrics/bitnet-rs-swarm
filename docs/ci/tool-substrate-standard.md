# Tool Substrate Standard

Status: proposed
Owner: release/ci
Created: 2026-06-03
Linked proposal:
[BITNET-PROP-0001: Proof Convergence and CI Economics](../proposals/BITNET-PROP-0001-proof-convergence-and-ci-economics.md)
Linked specs: n/a
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: n/a
Policy impact: Defines the preferred upstream engines that `xtask`, CI routing,
and policy ledgers should wrap before making them repo-facing contracts.

## Rule

BitNet-rs standardizes on a small upstream substrate and exposes repo-shaped
commands through `xtask` and policy ledgers.

```text
Do not make upstream tools the repo's public control surface.

Make xtask the repo surface.
Make upstream tools the engine room.
```

This keeps agents and CI jobs aligned with repository policy instead of asking
each caller to remember every upstream command, option, exception format, and
cost boundary.

## Standard stack

| Plane | Repo surface | Upstream engine room | Default posture |
| --- | --- | --- | --- |
| Repo orchestration | `cargo xtask ...` | Rust `xtask` commands | Public contract for agents, CI, and local proof. |
| Source exceptions | `cargo xtask allow-check`, `cargo xtask allow-diff` | `cargo-allow` | Ledgered exceptions with owner, reason, evidence, and review date. |
| Syntax and codemods | `cargo xtask policy-report`, targeted probes | `ast-grep`; rust-analyzer crates for Rust-authoritative identity | `ast-grep` finds candidates; Rust-aware tooling decides durable Rust identity. |
| Workspace graph | `cargo xtask ci plan`, release and risk-pack helpers | `cargo_metadata`, `guppy` | Cargo metadata for inventory; guppy-style graph queries for reverse dependencies, feature routing, and package partitioning. |
| Test execution | `cargo xtask test-pr`, `cargo xtask test-risk-pack`, `cargo xtask test-docs` | `cargo-nextest`, plus `cargo test --doc` | nextest for normal Rust tests; doctests stay explicit because they are a separate proof surface. |
| Coverage | `cargo xtask coverage` | `cargo-llvm-cov` | Execution-surface evidence only; not correctness, release readiness, or answer-quality proof. |
| Static mutation exposure | `cargo xtask ripr-pr` | `ripr` | PR-time weak-oracle and mutation-exposure signal for Rust behavior changes. |
| Runtime mutation | `cargo xtask mutation-targeted` | `cargo-mutants` | Targeted PR, nightly, and release backstop; not a default full-workspace PR tax. |
| Unsafe review and UB witnesses | `cargo xtask unsafe-review-pr`, `cargo xtask miri-targeted` | `unsafe-review`, Miri | Reviewability at PR time; concrete Miri witnesses only when targeted, nightly, or release policy asks for them. |
| Dependency trust | `cargo xtask check-deps`, `cargo xtask check-supply-chain` | `cargo-deny`, `cargo-vet`, RustSec / `cargo-audit`, `cargo-auditable` | `cargo-deny` is the normal gate; vet/audit/auditable provide maturity, advisory, and shipped-binary evidence. |
| Public API and release compatibility | `cargo xtask semver-check` | `cargo-semver-checks`, rustdoc JSON | Semver checks by default; rustdoc JSON only for custom API inventories and product-surface reports. |
| Workflow policy | `cargo xtask check-workflows` | `actionlint`, `zizmor` | actionlint for workflow correctness; zizmor for security posture, advisory first unless policy says blocking. |
| Text and config hygiene | `cargo xtask check-toml`, docs/text checks | `taplo`, `typos`, markdownlint/link tooling | Start with formatting, obvious structure, spelling, and broken-link hygiene; ratchet blocking scope only after dictionaries and baselines are stable. |
| Workspace hygiene | Scheduled/manual xtask lane | `cargo-udeps`; `cargo-hakari` only when duplicate-build pain is measured | Nightly/manual cleanup, not ordinary PR default. |
| CI cache economics | Workflow wrappers and policy ledgers | `Swatinem/rust-cache`; `sccache` only when economics justify it | Prefer cheap restore-on-PR behavior; avoid silent expensive hosted fallback. |

## Cost boundaries

Default PR routing should choose the cheapest truthful lane:

```text
docs/control-plane only -> no broad Rust compile
workflow-only -> workflow validation and normalized routed result only
Rust/build/test changes -> Rust-small lane + ripr where behavior changed
hardware or receipt-only -> syntax and receipt validation
unknown or mixed -> Rust-small, not full CI
```

Heavy lanes require an explicit trigger from labels, manual dispatch, schedules,
release prep, merge queue policy, or a risk-pack rule. In particular:

- full mutation is not a default PR gate;
- full Miri is not a default PR gate;
- coverage is not a correctness claim;
- model downloads, hardware lanes, release lanes, and full-platform matrices are
  not substitutes for scoped PR proof;
- missing self-hosted runner capacity must not silently become a long
  GitHub-hosted full-CI fallback.

## Candidate versus authority

Some engines are excellent candidate generators but are not authoritative for
all repo policy decisions.

```text
ast-grep finds syntactic candidates.
rust-analyzer crates preserve Rust source identity where policy must survive
refactors.
git ls-files defines the tracked source inventory for file-policy scans.
cargo_metadata and guppy understand workspace/package/target relationships.
xtask converts those facts into repository policy decisions.
```

Use ignored files, generated files, or filesystem walks only when the specific
lane intentionally scans beyond tracked source state.

## Expected stable xtask surface

`xtask` should expose stable commands even when the implementation behind them
changes:

```bash
cargo xtask check-pr
cargo xtask fix-pr
cargo xtask pr-summary

cargo xtask allow-check
cargo xtask allow-diff
cargo xtask ripr-pr
cargo xtask unsafe-review-pr

cargo xtask test-pr
cargo xtask coverage
cargo xtask mutation-targeted
cargo xtask miri-targeted

cargo xtask check-deps
cargo xtask check-supply-chain
cargo xtask semver-check
cargo xtask check-workflows
cargo xtask check-toml
cargo xtask policy-report
```

Each command may be advisory, blocking, scheduled, or release-only according to
lane policy. The command name is the repo contract; the upstream binary is an
implementation detail unless a proof receipt explicitly records the exact tool
version and invocation.

## Adoption rules

1. Add or change an upstream tool behind an `xtask` command or policy ledger
   first, then wire CI to that repo-facing surface.
2. Keep exception state in ledgers such as `policy/*.toml`; do not bury durable
   exceptions in workflow YAML, shell conditionals, or chat history.
3. Use `git ls-files -z` as the normal source inventory for policy scans.
4. Preserve heavy-workflow no-cancel semantics unless the workflow explicitly
   documents itself as safe to cancel.
5. Upload large receipts on failure or explicit evidence lanes, not by default
   for metadata-only PRs.
6. Record unavailable tools as unavailable evidence; do not summarize skipped
   proof as a passing check.
