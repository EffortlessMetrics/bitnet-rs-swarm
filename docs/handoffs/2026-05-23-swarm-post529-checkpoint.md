# Swarm Post-529 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #524, #525, #526, #527, #528, #529

This file records the swarm/source state at swarm SHA
`eba421a87468344ff5a03c2827e72b481a9b9444`, after the
`LNL258V-GOAL-AUDIT-051` tracker closeout returned the queue to zero open work,
excluding this handoff PR. PR #529 is the current `origin/main` tip.

## Landed Since Post-523

- #524 `docs(handoffs): record swarm post-523 checkpoint`
  - Squash commit: `cab6d1b69f958b5df1042321598de19ccf2b25b2`
  - Recorded the prior post-523 queue checkpoint.
- #525 `CUDA-BITNET-PERF-005: build strict BitNet repeated profile receipts`
  - Squash commit: `32cca5961a3e3905c7a091bfb34dfada2649102e`
  - Added strict CUDA BitNet repeated-profile receipt aggregation tooling.
  - Claim boundary: receipt tooling only. No Lunar Lake route evidence, CUDA
    route promotion, Qwen3 promotion, Arc/NPU promotion, BitNet acceleration
    promotion, speedup, residency, quality, or support claim.
- #526 `docs(tracking): LNL258V-GOAL-AUDIT-051 post-A770/CUDA refresh`
  - Squash commit: `c6a60490756ffa3eb2a57dcf138f8dfd354116a0`
  - Refreshed the no-inference Lunar Lake audit/checklist through #525.
  - Claim boundary: audit/tracker refresh only. No Lunar Lake inference, route
    promotion, fallback behavior change, speedup, power advantage, battery-mode
    evidence, measured-temperature evidence, native accelerator proof, CPU/A770
    parity, strict A770 readiness, trusted partial acceleration, full BitNet
    accelerator inference, or BitNet QK256/I2_S behavior claim.
- #528 `test(cuda): remove no-panic debt from repeated profile tests`
  - Squash commit: `c69ffb24e220d331a51b071c327b7c1e3a541422`
  - Removed no-panic policy debt from CUDA repeated-profile tests.
  - Claim boundary: test/policy cleanup only. No new benchmark, speedup,
    readiness, support, route promotion, or hardware claim.
- #529 `docs(tracking): close LNL258V-GOAL-AUDIT-051`
  - Squash commit and current queue-stabilized swarm SHA:
    `eba421a87468344ff5a03c2827e72b481a9b9444`
  - Closed `LNL258V-GOAL-AUDIT-051`, added the merged event ledger entry, and
    regenerated dashboards so the Intel 258V lane returned to
    `LNL258V-POWER-006`.
  - Claim boundary: tracker closeout only. No Lunar Lake inference, route
    promotion, speedup, power advantage, battery-mode evidence, measured
    temperature, native accelerator proof, Qwen3 promotion, CPU/A770 parity,
    strict A770 readiness, trusted partial acceleration, full BitNet accelerator
    inference, or BitNet QK256/I2_S behavior claim.

## Closed Unmerged

- #527 `docs(a770): close A770-041`
  - Closed at `2026-05-23T22:55:54Z` without merge.
  - Disposition: superseded by #523, which already closed A770-041 on
    `origin/main` with squash commit
    `1480cdf778193499f9f1baab9c7c2d9c1fe667e3`.

## Open Queue After #529

- No open PRs in `EffortlessMetrics/bitnet-rs-swarm`, excluding this handoff
  PR.
- No open PRs in `EffortlessMetrics/BitNet-rs`.

## Source Delta At This Checkpoint

- Checkpoint source ref:
  `source/main` = `ef6eec8a6f95a54138fd69617235347944d2caae`
- Checkpoint swarm ref:
  `origin/main` = `eba421a87468344ff5a03c2827e72b481a9b9444`
- `source/main` is reachable from `origin/main`.
- `rtk git rev-list --left-right --count source/main...origin/main`
  returned `0 143`.
- `rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust4/bitnet-rs-swarm && cargo run --locked -p xtask --no-default-features -- repo-boundary status --source-ref source/main --swarm-ref origin/main'`
  returned `repo-boundary status: Ok`.
- Release workflow guard status: guarded.
- Normalized result check: `BitNet Rust Small Result`.

## Promotion Packet Readiness

Do not promote `source/main..origin/main` as one source PR from this checkpoint
alone. The range now spans 143 swarm-only commits across repo-boundary tooling,
tracker closeouts, CUDA/Qwen3 receipt evidence, SLM CPU boundary work, A770
diagnostic lineage, Lunar Lake audit state, generated dashboards, and runtime
diagnostic code.

Use `eba421a87468344ff5a03c2827e72b481a9b9444` as the current swarm promotion
candidate SHA only after the source-promotion operator chooses a bounded batch
and verifies source-owned release, signing, publish, workflow, and secrets-heavy
surfaces.

## Evidence

- Open swarm PR queue:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,statusCheckRollup,updatedAt`
  returned `[]` before this handoff PR was opened.
- Open source PR queue:
  `rtk gh pr list --repo EffortlessMetrics/BitNet-rs --state open --limit 50 --json number,title,headRefName,mergeStateStatus,updatedAt`
  returned `[]`.
- Current swarm main:
  `rtk git rev-parse origin/main` returned
  `eba421a87468344ff5a03c2827e72b481a9b9444`.
- Current source main:
  `rtk git rev-parse source/main` returned
  `ef6eec8a6f95a54138fd69617235347944d2caae`.
- Source/swarm ancestry:
  `rtk git merge-base --is-ancestor source/main origin/main` returned success.
- Source/swarm delta:
  `rtk git rev-list --left-right --count source/main...origin/main` returned
  `0 143`.
- Repo-boundary status:
  `rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust4/bitnet-rs-swarm && cargo run --locked -p xtask --no-default-features -- repo-boundary status --source-ref source/main --swarm-ref origin/main'`
  returned status `Ok`, source missing commits `0`, swarm-only commits `143`,
  `release_workflows_guarded: true`, and normalized result
  `BitNet Rust Small Result`.
- #529 auto-merged by squash at
  `eba421a87468344ff5a03c2827e72b481a9b9444` after the generated tracker
  closeout, `Doctor + Generated Dashboards`, and normalized
  `BitNet Rust Small Result` completed.

## Validation

- `rtk git diff --check`
- `rtk proxy npx --yes markdownlint-cli2 "docs/handoffs/2026-05-23-swarm-post529-checkpoint.md"`
- `rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust4/bitnet-rs-swarm && cargo run --locked -p xtask --no-default-features -- repo-boundary status --source-ref source/main --swarm-ref origin/main'`

## Remaining Work

- Continue processing new swarm PRs as they arrive: classify lane, inspect exact
  diff, verify claim boundaries, run focused proof, and squash-merge normal
  swarm PRs only when green.
- Keep source-history repair, source-to-swarm sync, and swarm-to-source
  promotion ancestry-preserving. Do not squash those repository-boundary PRs.
- Keep generated dashboards, campaign `active.toml`, workflows, `Cargo.lock`,
  root policy files, and authority docs under short merge-window discipline.
- Treat A770/BitNet diagnostics as a salvage lane. Port durable tools, tests,
  reports, and claim gates as small successors; close transient probes only
  after useful learning is preserved.

## Blockers

- None for the current handoff checkpoint.
- Source promotion remains a separate, ancestry-preserving source-owner
  operation.

## Claim Boundary

This handoff records operator state only. It does not claim A770 support, CUDA
support, Apple M4 support, Lunar Lake readiness, SLM CPU release readiness,
quality, speed, server readiness, selected attention, resident KV cache, full
residency, reference parity, source release readiness, source-to-swarm sync
completion, or swarm-to-source promotion completion.

## Next Operator Commands

```powershell
rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt,statusCheckRollup
rtk gh pr list --repo EffortlessMetrics/BitNet-rs --state open --json number,title,headRefName,mergeStateStatus,updatedAt,statusCheckRollup
rtk git fetch origin --prune
rtk git fetch source --prune
rtk git rev-parse origin/main
rtk git rev-parse source/main
rtk git merge-base --is-ancestor source/main origin/main
rtk git rev-list --left-right --count source/main...origin/main
rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust4/bitnet-rs-swarm && cargo run --locked -p xtask --no-default-features -- repo-boundary status --source-ref source/main --swarm-ref origin/main'
```
