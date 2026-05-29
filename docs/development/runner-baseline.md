# BitNet Swarm Runner Baseline

First evidence of the `EM CI Routed Rust` lane on `EffortlessMetrics/bitnet-rs-swarm`.

## Workflow

`.github/workflows/em-ci-routed-rust.yml` routes the small BitNet Rust lane.

> **Self-hosted-only update (BITNET-ADR-0008).** The GitHub-hosted fallback lane
> has been removed. Coordination jobs (route, result) now run on
> `[self-hosted, linux, x64]` rather than `ubuntu-latest`, since this swarm repo
> is self-hosted-only. The diagram below reflects the current routing.

```
Route BitNet Rust Small  (self-hosted linux x64, gh api orgs/.../actions/runners)
  ├─ target=cx53     → BitNet Rust Small on CX53   (self-hosted, docker em-ci-rust:1.95)
  ├─ target=cx43     → BitNet Rust Small on CX43   (self-hosted, docker em-ci-rust:1.95)
  ├─ target=blocked  → no idle trusted runner; normalized result fails closed
  └─ fork PR         → blocked (fork_pr_self_hosted_untrusted); never self-hosted

BitNet Rust Small Result  (self-hosted linux x64, normalized gate)
```

The router selects CX53 when an org runner with labels
`em-ci, cx53, rust-small, trusted-pr` is `online` and `busy=false`.
If no CX53 runner is idle, it selects CX43 when an org runner with labels
`em-ci, cx43, rust-small, trusted-pr` is `online` and `busy=false`.
There is no hosted fallback: when no trusted runner is idle the router emits
`target=blocked` / `reason=no_idle_runner` and the normalized result fails
closed. Fork PRs are blocked from self-hosted execution
(`fork_pr_self_hosted_untrusted`).

## First-run evidence (2026-05-17/18)

### PR-path run (PR #1, attempt 2)
- Run: `26006239077`
- Route: success, 5s, `router_target=cx53`, `router_reason=cx53_idle`
- CX53: success, 6m25s, runner `em-ci-hel2-cx53-rust-01` (cold sccache)
- GitHub Hosted: skipped
- Result: success

### Workflow-dispatch on `main`
- Run: `26009235657`
- Route: success, 5s, `router_target=cx53`, `router_reason=cx53_idle`
- CX53: success, 3m14s, runner `em-ci-hel2-cx53-rust-01` (warm sccache)
- GitHub Hosted: skipped
- Result: success

Warm sccache cut CX53 lane runtime from 6m25s to 3m14s.

## Self-hosted runner labels

CX53: `em-ci, cx53, rust-small, trusted-pr` on group `em-ci-small`.
CX43: `em-ci, cx43, rust-small, trusted-pr` on group `em-ci-small`.
Container image: `em-ci-rust:1.95`.
Persistent mounts: `/mnt/ci-cache/{cargo-home,sccache}`, ephemeral
`/mnt/ci-scratch/{tmp,target}/<run>-<attempt>` cleaned in the same job.

## Branch protection (planned, not yet enabled)

When enabled, branch protection on `main` should require **only**
`BitNet Rust Small Result`. Do not require the conditional implementation
lanes (`Route BitNet Rust Small`, `BitNet Rust Small on CX53`,
`BitNet Rust Small on CX43`) — they are not always present in a single run.
The GitHub-hosted fallback lane has been removed (BITNET-ADR-0008).

## What is intentionally not yet wired

- Model-cache lane (no `HF_TOKEN`, no `/mnt/model-cache` mount).
- CX33 BitNet backfill runner.
- Branch protection (pending tiny-PR validation of the same-repo PR path).
- Public `EffortlessMetrics/BitNet-rs` cutover (queue drain pending).
