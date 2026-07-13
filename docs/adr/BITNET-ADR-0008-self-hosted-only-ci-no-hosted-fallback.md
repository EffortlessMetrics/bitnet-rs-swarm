# BITNET-ADR-0008: Self-Hosted-Only CI With No GitHub-Hosted Fallback

- **Status:** Superseded by BITNET-ADR-0011
- **Date:** 2026-05-29
- **Linked proposal/spec:** n/a (CI economics / runner policy decision)
- **Linked plan:** [swarm-runner-rollout-plan.md](../development/swarm-runner-rollout-plan.md),
  [runner-baseline.md](../development/runner-baseline.md)
- **Superseded by:** [BITNET-ADR-0011](./BITNET-ADR-0011-lean-opt-in-github-hosted-fallback.md)
- **Supersedes (for `bitnet-rs-swarm` only):** the "hosted fallback preserved"
  and "release/publish/signing workflows touched" stop conditions of the swarm
  runner rollout plan.

> This ADR is retained as historical context for the self-hosted-only period.
> The current routing and fallback policy is defined by BITNET-ADR-0011.

## Context

`bitnet-rs-swarm` is a public, trusted same-repo swarm CI repository. Its
workflows previously mixed GitHub-hosted runners (`ubuntu-latest`,
`ubuntu-22.04`, `macos-14`, `windows-latest`) with a self-hosted routing layer
(`CX53 -> CX43 -> GitHub-hosted`) for the EM routed Rust lane. The hosted
fallback was reachable via the `allow-github-hosted` / `ci-budget-ack` PR
labels.

Mixed routing has two costs that the rollout plan was explicitly trying to
control: accidental GitHub-hosted minutes (budget leakage) and ambiguous LEM
accounting, because the same logical lane could run on a hosted or a
self-hosted runner depending on label state and runner idleness.

The org runs a self-hosted fleet under group `em-ci-small`. **As of this
decision the fleet provides only `self-hosted, linux, x64` capacity** (the
`em-ci` `cx53` / `cx43` runners). Runners for `macOS` (x64/ARM64),
`windows, x64`, `linux, ARM64`, and GPU-specific labels are **not yet
provisioned.**

## Decision

For `bitnet-rs-swarm`, CI runs **only** on self-hosted runners. There is **no
GitHub-hosted fallback**:

1. Every `runs-on` in `.github/workflows/*` targets a self-hosted label array
   (e.g. `[self-hosted, linux, x64]`, `[self-hosted, macOS, ARM64]`). No
   `ubuntu-*`, `macos-*`, or `windows-*` hosted aliases remain.
2. The EM routed Rust router (`em-ci-routed-rust.yml`) drops the
   `rust-small-github` fallback job and the `allow-github-hosted` /
   `ci-budget-ack` escape hatch. Because there is no hosted fallback to absorb
   load, the router **queues rather than fails on a busy fleet**: it routes to
   an idle `cx53`/`cx43` runner when one exists, otherwise to a busy-but-online
   `cx43`/`cx53` pool (reason `cx43_busy_queued` / `cx53_busy_queued`) so GitHub
   queues the job until a runner frees. It emits `blocked` (`no_online_runner`)
   only when no trusted runner is online at all — a genuine outage, not load.
3. Fork PRs, a missing/invalid `EM_RUNNER_READ_TOKEN`, and runner-API/parse
   failures still fail closed (`blocked` / `fork_pr_self_hosted_untrusted`,
   `token_missing`, `runner_api_failed`, `parse_failed`) instead of routing to a
   hosted runner.
4. `policy/ci-lane-whitelist.toml` runner multipliers and lane runner keys are
   expressed in self-hosted identities (`self_hosted_linux_x64`,
   `self_hosted_windows_x64`, `self_hosted_macos_arm64`, `self_hosted_gpu`).

## Consequences

Positive:

- No accidental GitHub-hosted minutes; CI budget and LEM accounting map to a
  single, explicit set of self-hosted runner identities.
- Routing, budgeting, and trust boundaries are explicit: fork/untrusted work is
  blocked rather than silently downgraded to hosted.

Negative / operational constraints (must be respected while the fleet is
linux-x64-only):

- Lanes whose `runs-on` requires hardware the fleet does not yet have
  (`macOS`, `windows`, `linux ARM64`, GPU) are **dormant**: a job scheduled on
  a missing label sits queued indefinitely. Such lanes **must not** be required
  branch-protection checks and **must not** start on an ordinary
  `pull_request` / `merge_group` run. They run only when explicitly opted in
  (PR label such as `macos` / `apple-silicon` / `metal` / `gpu` / `full-ci`,
  `workflow_dispatch`, `schedule`, or tag/release events) and stay non-blocking
  until the corresponding runners are provisioned.
- The cross-platform `release.yml` matrix cannot produce non-linux artifacts
  until the matching self-hosted runners exist. Release builds on those targets
  are deferred, not silently moved to hosted.
- Only `linux x64` PR lanes (build/test, clippy, docs, receipt/policy gates,
  and the normalized `BitNet Rust Small Result`) are live and blocking.

## Claim boundary

This ADR decides **CI runner routing and budget policy only**. It does not:

- claim the non-linux self-hosted fleet exists or is validated;
- prove any backend, kernel, tokenizer, or model behavior;
- change branch-protection required checks (still exactly the normalized
  results per the rollout plan) — that is a separate, deferred change.

## Alternatives considered

- **Keep the documented `… -> GitHub-hosted` fallback chain.** Rejected: the
  fallback is the source of the budget leakage and ambiguous LEM accounting this
  decision exists to remove.
- **Incremental per-lane routing PRs only (leave release/GPU/matrix hosted).**
  Reasonable and lower-risk, but retains hosted minutes on the long-tail lanes
  indefinitely; rejected in favor of a single explicit self-hosted-only
  surface with dormant non-linux lanes.

## How to revert

- Restore `rust-small-github` and the `allow-github-hosted` / `ci-budget-ack`
  label gate in `em-ci-routed-rust.yml`, and re-point each `runs-on` back to the
  hosted alias it replaced (the pre-migration values are recoverable from the
  migration commit). Restore the hosted runner multiplier keys in
  `policy/ci-lane-whitelist.toml`. Mark this ADR `Superseded by NNNN`.
