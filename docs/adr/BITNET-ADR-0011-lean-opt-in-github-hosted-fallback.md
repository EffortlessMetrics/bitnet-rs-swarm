# BITNET-ADR-0011: Lean Opt-In GitHub-Hosted Rust Fallback

- **Status:** Accepted
- **Date:** 2026-07-12
- **Linked proposal/spec:** n/a (CI economics / runner policy decision)
- **Linked plan:** [swarm-runner-rollout-plan.md](../development/swarm-runner-rollout-plan.md)
- **Supersedes:** [BITNET-ADR-0008](./BITNET-ADR-0008-self-hosted-only-ci-no-hosted-fallback.md)

## Context

The self-hosted `em-ci-small` fleet is the preferred execution surface for
`bitnet-rs-swarm`, but an absent fleet can leave the normalized Rust-small
check queued indefinitely. A hosted fallback is useful for short-lived
continuity, but an automatic hosted equivalent would turn an infrastructure
outage into uncontrolled GitHub Actions spend.

## Decision

Keep self-hosted routing first and restore exactly one bounded hosted fallback
for the normalized `BitNet Rust Small Result` lane:

1. The router selects `cx53` or `cx43` whenever a matching trusted runner is
   online. A busy-but-online pool remains queued; it does not spill to hosted.
2. The router may select `github_hosted` only when no trusted self-hosted runner
   is online and the caller explicitly authorizes the spend with
   `allow-github-hosted`, `full-ci`, or `ci-budget-ack`. Workflow dispatch may
   use the boolean `allow_github_hosted` input. The separate
   `ci-budget-override` label may select the same bounded hosted proof even
   when a runner is online but known to be unhealthy; this is an explicit
   recovery override, not a default spillover path.
3. The fallback is same-repository only. Fork pull requests remain blocked from
   self-hosted and hosted Rust execution.
4. The fallback runs only the lean Rust-small proof on pinned `ubuntu-22.04`:
   no Docker, model downloads, credentials, GPU, hardware, broad feature
   matrix, coverage, fuzzing, performance, or release work.
5. The normalized `BitNet Rust Small Result` remains the only routed Rust check
   intended for branch protection. Conditional implementation jobs may be
   skipped when another target is selected.
6. Heavy or broad workflows retain their documented `cancel-in-progress: false`
   semantics. This fallback is short and opt-in; it is not a general hosted
   replacement for self-hosted or hardware lanes.

## Consequences

Positive:

- Ordinary PRs remain self-hosted-only unless an explicit budget decision is
  made.
- A missing self-hosted fleet can be recovered with a small, predictable proof.
- The fallback cannot silently run models, Docker, GPU, hardware, or full CI.
- The route and normalized result retain one stable branch-protection surface.

Negative:

- A maintainer must apply an authorization label or dispatch input during a
  self-hosted outage; `ci-budget-override` is additionally required when the
  online fleet is known to be unhealthy.
- Hosted fallback results are not equivalent to the self-hosted Docker image;
  they prove the lean Rust-small package/test surface only.

## Claim boundary

This ADR governs CI runner routing and budget behavior only. It does not prove
BitNet model quality, Apple/M4 behavior, GPU execution, release readiness, or
hosted/self-hosted performance parity.

## How to revert

Remove the `github_hosted` target and hosted job from
`.github/workflows/em-ci-routed-rust.yml`, restore the no-fallback language in
the runner baseline, and mark this ADR superseded by the replacement decision.
