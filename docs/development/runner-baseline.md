# BitNet Swarm Runner Baseline

Current baseline for the `EM CI Routed Rust` lane on
`EffortlessMetrics/bitnet-rs-swarm`.

## Workflow

`.github/workflows/em-ci-routed-rust.yml` routes the small BitNet Rust lane.

The self-hosted fleet remains the preferred route. The bounded hosted fallback
decision is [BITNET-ADR-0011](../adr/BITNET-ADR-0011-lean-opt-in-github-hosted-fallback.md).

```
Route BitNet Rust Small  (ubuntu-latest coordination, org runner discovery)
  ├─ target=cx53          → BitNet Rust Small on CX53   (self-hosted Docker)
  ├─ target=cx43          → BitNet Rust Small on CX43   (self-hosted Docker)
  ├─ target=github_hosted → lean Rust-small proof       (ubuntu-22.04, opt-in)
  ├─ target=blocked       → no runner and no fallback authorization
  └─ fork PR              → blocked from self-hosted and hosted fallback

BitNet Rust Small Result  (ubuntu-latest, normalized gate)
```

The router prefers an idle runner: it selects CX53 when an org runner with
labels `em-ci, cx53, rust-small, trusted-pr` is `online` and `busy=false`, else
CX43 on the same basis. **When no runner is idle but the cx43/cx53 pool is still
online (just busy), the router routes to that pool anyway** (reason
`cx43_busy_queued` / `cx53_busy_queued`) and lets GitHub queue the job until a
runner frees — it never spends hosted minutes merely because the fleet is busy.
Only when no trusted self-hosted runner is online does it select
`github_hosted`, and only when the PR has `allow-github-hosted`, `full-ci`, or
`ci-budget-ack`, or a workflow dispatch explicitly sets
`allow_github_hosted=true`. The `ci-budget-override` label is a separate
explicit recovery path for an online runner known to be unhealthy. The hosted
job runs no Docker, models, credentials, GPU, hardware, or broad feature matrix.

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
`BitNet Rust Small on CX43`, `BitNet Rust Small on GitHub Hosted`) — they are
not always present in a single run.

## What is intentionally not yet wired

- Model-cache lane (no `HF_TOKEN`, no `/mnt/model-cache` mount).
- CX33 BitNet backfill runner.
- Branch protection (pending tiny-PR validation of the same-repo PR path).
- Public `EffortlessMetrics/BitNet-rs` cutover (queue drain pending).
