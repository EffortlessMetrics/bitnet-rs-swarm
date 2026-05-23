# Swarm Post-502 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #496, #498, #499, #500, #501, #502

## Landed

- #496 `xtask(repo-boundary): report checkout role`
  - Squash commit: `bd30a5507986585d4ff11bb64b8772d99f754dfa`
  - Added checkout-role reporting to the repo-boundary status command.
  - Claim boundary: repository-boundary/status tooling only; no runtime,
    model, receipt, hardware, CI routing, release, publish, signing, or source
    promotion behavior change.
- #498 `LNL258V-GOAL-AUDIT-047: refresh audit after A770-038`
  - Squash commit: `784dc0330316af6bc8fdce3c502f0030ee263369`
  - Refreshed the no-inference Lunar Lake audit/checklist after A770-038.
  - Claim boundary: no Lunar Lake inference, model load, route promotion,
    fallback, speedup, power, battery, native accelerator, Qwen3, CPU-A770
    parity, or BitNet QK256 claim.
- #499 `LNL258V-GOAL-AUDIT-047: close tracker after PR 498`
  - Squash commit: `84249f103a8537e02d772293c7b14457b222cdb3`
  - Closed the Lunar Lake audit tracker state after #498 and regenerated
    dashboards.
  - Claim boundary: tracker closeout only.
- #501 `feat(cuda): add qwen3 decode-128 run 2 receipt`
  - Squash commit: `f8f887bd2eda6e7d00711b9021d9c4497c877631`
  - Added a current-source Qwen3 decode-128 strict CUDA receipt.
  - Claim boundary: receipt evidence only; no aggregate completion, speedup,
    full residency, broad dense GGUF readiness, Qwen2.5 inheritance, BitNet
    packed I2_S/QK256 proof, or server readiness claim.
- #500 `diag(a770): add earlier block source frontier`
  - Squash commit: `75cb05cd7f24467d326f61dda2c8a1848beafac4`
  - Added A770-039 earlier-block source-frontier diagnostic tooling, reports,
    and receipts.
  - Claim boundary: diagnostic-only. No runtime math, dispatch, kernel,
    scoring, sampling, CPU/A770 parity, reference parity, answer readiness,
    quality, residency, speed, trusted partial acceleration, or full BitNet
    inference claim.
- #502 `A770-039: close tracker after PR 500`
  - Squash commit and current queue-stabilized swarm SHA:
    `d4bceb9b880b9140deaa6eadd8271deaf2e551a5`
  - Closed the A770-039 tracker state after #500.
  - Claim boundary: tracker closeout only.

## Closed Unmerged

- None in this checkpoint window.

## Open Queue After #502

- No open PRs in `EffortlessMetrics/bitnet-rs-swarm`.
- No open PRs in `EffortlessMetrics/BitNet-rs`.

## Source Delta

- Current source ref:
  `origin/main` = `ef6eec8a6f95a54138fd69617235347944d2caae`
- Current swarm ref:
  `swarm/main` = `d4bceb9b880b9140deaa6eadd8271deaf2e551a5`
- `origin/main` is reachable from `swarm/main`.
- `rtk git rev-list --left-right --count origin/main...swarm/main`
  returned `0 117`.
- `cargo run --locked -p xtask --no-default-features -- repo-boundary status
  --source-ref origin/main --swarm-ref swarm/main` returned
  `repo-boundary status: Ok`.
- Release workflow guard status: guarded.
- Normalized result check: `BitNet Rust Small Result`.

## Promotion Packet Readiness

A dry-run packet was generated under `target/promotion/` for:

```text
origin/main..swarm/main
ef6eec8a6f95a54138fd69617235347944d2caae..d4bceb9b880b9140deaa6eadd8271deaf2e551a5
```

The generated range is intentionally not committed as a promotion packet. It
spans 117 swarm-only commits, five campaigns, runtime/product code, workflows,
hardware receipts, generated dashboards, repo-boundary tooling, A770
diagnostics, SLM CPU diagnostics, Qwen3 CUDA receipt work, and Lunar Lake audit
state. That is too broad for one source promotion.

Recommended promotion split:

- repo-boundary/control-plane docs and xtask tooling;
- tracker-only closeouts and generated dashboard state;
- Qwen3 CUDA receipt evidence, preserving exact non-promotion boundaries;
- SLM CPU diagnostic/storage-boundary slices;
- A770 diagnostic lineage, preserving diagnostic-only claim boundaries;
- Lunar Lake audit/checklist refreshes.

Each source promotion packet should name its included swarm PRs, proof commands,
receipts, generated artifacts, claim boundary, excluded work, release impact,
and rollback independently.

## Evidence

- Open swarm PR queue:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --limit 50 --json number,title,isDraft,mergeStateStatus,updatedAt,headRefName,baseRefName,labels,url`
  returned `[]`.
- Open source PR queue:
  `rtk gh pr list --repo EffortlessMetrics/BitNet-rs --state open --limit 50 --json number,title,isDraft,mergeStateStatus,updatedAt,headRefName,baseRefName,labels,url`
  returned `[]`.
- Repo-boundary status:
  `rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust/BitNet-rs && cargo run --locked -p xtask --no-default-features -- repo-boundary status --source-ref origin/main --swarm-ref swarm/main'`
  returned status `Ok`, source missing commits `0`, swarm-only commits `117`,
  and `release_workflows_guarded: true`.
- Dry-run promotion packet generation:
  `rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust/BitNet-rs && cargo run --locked -p xtask --no-default-features -- promote-to-source --from origin/main --to swarm/main --out target/promotion/swarm-through-d4bceb9b8.md'`
  completed and wrote the packet under `target/promotion/`.

## Validation

- #500 local proof before branch refresh/merge:
  - `cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli earlier_block_source -- --nocapture`
  - `cargo check --locked -p bitnet-transformer --no-default-features --features cpu`
  - `cargo check --locked -p bitnet-models --no-default-features`
  - `cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli,opencl`
  - `rustfmt --edition 2024 --check crates/bitnet-transformer/src/lib.rs crates/bitnet-models/src/bitnet.rs crates/bitnet-cli/src/main.rs crates/bitnet-cli/src/commands/answer_parity.rs`
  - `cargo run --locked -p xtask --no-default-features -- campaign check intel-a770`
  - `cargo run --locked -p xtask --no-default-features -- campaign generate --check`
  - `cargo run --locked -p xtask --no-default-features -- campaign doctor`
  - `git diff --check`
- #500 remote proof:
  - `BitNet Rust Small Result`: `SUCCESS`
  - `PR Gate Success`: `SUCCESS`
  - campaign tracker, guards, docs, feature-matrix, validation, CI core, and
    related required checks completed successfully before merge.
- #502 remote proof:
  - `BitNet Rust Small Result`: `SUCCESS`
  - `CI Core Success`: `SUCCESS`
  - tracker closeout PR merged after the normalized result passed.
- Validation gap: source promotion proof has not run. This checkpoint records
  swarm state and promotion-split guidance only.

## Remaining Work

- Do not promote `origin/main..swarm/main` as one source PR.
- Prepare smaller promotion packets by lane or source-risk group before opening
  `BitNet-rs` source PRs.
- Keep processing new swarm PRs as they arrive: classify lane, inspect exact
  diff, verify claim boundaries, run focused proof, and squash-merge normal
  swarm PRs only when green.
- Keep source-history repair, source-to-swarm sync, and swarm-to-source
  promotion ancestry-preserving. Do not squash those repository-boundary PRs.
- Keep generated dashboards, campaign `active.toml`, workflows, `Cargo.lock`,
  root policy files, and authority docs under short merge-window discipline.

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
rtk git fetch swarm --prune
rtk git rev-parse origin/main
rtk git rev-parse swarm/main
rtk git merge-base --is-ancestor origin/main swarm/main
rtk git rev-list --left-right --count origin/main...swarm/main
rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust/BitNet-rs && cargo run --locked -p xtask --no-default-features -- repo-boundary status --source-ref origin/main --swarm-ref swarm/main'
rtk wsl -d Ubuntu -- bash -lc 'cd /mnt/h/Code/Rust/BitNet-rs && cargo run --locked -p xtask --no-default-features -- promote-to-source --from origin/main --to swarm/main --out target/promotion/swarm-through-$(git rev-parse --short=12 swarm/main).md'
```
