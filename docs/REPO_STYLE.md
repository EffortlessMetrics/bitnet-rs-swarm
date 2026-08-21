# Repo style

BitNet-rs is operated as an evidence machine: strict defaults, owned
exceptions, static signal first, runtime proof where it pays, receipts
everywhere, and one review-fast PR at a time.

Rust and `xtask` are the default construction material. Non-Rust files,
unsafe, panic paths, lint suppressions, generated files, workflow behavior,
process/network access, expensive CI lanes, and release claims must be owned
and receipted.

## Tool roles

The durable model is deliberately consolidated. `xtask` is the repo control
plane; it wraps upstream tools, aggregates receipts, and enforces BitNet-rs
policy glue. It must not grow into a second implementation of every upstream
checker.

| Tool | Repo role |
| --- | --- |
| `cargo-allow` | Durable source-exception ledger for visible exceptions. |
| `ripr` | Static mutation-exposure analysis for weak-oracle signal shifted left. |
| `unsafe-review` | Unsafe-contract reviewability for changed unsafe seams. |
| `xtask` | Orchestration, receipts, repo-local policy, CI planning, and release control. |
| `cargo-mutants` | Runtime mutation backstop where static signal or risk warrants it. |
| Miri | Concrete UB execution backstop for selected witness routes. |
| Codecov | Execution-surface telemetry, not a substitute for assertions. |

## Static evidence first

Static evidence runs before expensive runtime proof whenever it can answer the
review question cheaply:

- `cargo-allow` owns source exceptions and prevents invisible retained debt;
- `ripr` reports static mutation-exposure and weak-oracle risk;
- `unsafe-review` checks whether unsafe seams are reviewable;
- rustc and Clippy enforce code-shape policy and suppression governance.

Static evidence is not a correctness proof. It is the first high-signal filter
that tells reviewers where focused runtime proof may pay.

## Runtime evidence where it pays

Runtime evidence is preserved, but routed by risk:

- focused tests on ordinary PRs;
- targeted mutation for risk PRs or high-value surfaces;
- Miri, fuzzing, broader mutation, coverage, and release readiness on labels,
  schedules, main, campaign lanes, or release lanes.

Do not move deep validation out of the system. Move it to the lane where it
produces useful evidence per Linux-equivalent minute.

## Exception rule

There are no invisible source exceptions. Unsafe blocks, panic-family calls,
lint suppressions, generated files, workflow behavior, non-Rust automation,
process/network access, and release claims need an owner, reason, evidence, and
review path through `cargo-allow`, policy TOML, or the relevant review tool.

Bad exception reasons are words like `legacy`, `misc`, `needed`, or
`temporary`. Good reasons identify the boundary, why the exception remains,
what covers it, and when it will be reviewed or removed.

## CI economics

CI is designed for proof per Linux-equivalent minute (LEM), not fewer checks.
Default PRs should be cheap, deterministic, and high-signal. Deep validation
must stay available, but it should be selected by changed risk surface, label,
main, nightly, release, or campaign proof requirements.

A skipped optional lane is not a pass. It is a policy decision that must remain
visible in receipts or summaries.

## Agent working model

Agents work one review-fast PR at a time. Review-fast does not mean tiny; it
means a coherent seam, nearby proof, efficient verification, and an honest claim
boundary.

For each PR:

1. inspect current status and the linked source-of-truth artifacts;
2. keep scope to one behavior, seam, policy slice, or documentation doctrine;
3. avoid unrelated cleanup and invisible exceptions;
4. run the selected proof plus `git diff --check`;
5. record what the PR proves, what it does not prove, and the follow-up lane;
6. clean up temporary work after the PR lands.

Do not broaden scope to satisfy CI. Do not add broad suppressions. Do not add
shell, Python, or JavaScript repo automation when Rust/`xtask` is the durable
home unless the non-Rust surface is owned and receipted.
