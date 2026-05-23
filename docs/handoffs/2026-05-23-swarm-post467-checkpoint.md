# Swarm Post-467 Checkpoint Handoff

Status: queue checkpoint, no source promotion started
Linked proposal: not changed by this handoff
Linked specs: not changed by this handoff
Linked ADRs: not changed by this handoff
Linked plan: post-repair swarm operating phase
Campaign: swarm merge marshal / repo operating phase
PRs: #459, #464, #465, #463, #466, #467

## Landed

- #459 `perf(slm): SLM-CPU-088 residual block output boundary`
  - Squash commit:
    `c1050aa483bac674c26b523dd7b6182a40801827`
  - SLM runtime/perf boundary instrumentation only.
  - Claim boundary: no release readiness, source promotion, broad quality,
    hardware readiness, or public support claim.
- #464 `docs(slm): close SLM-CPU-088`
  - Squash commit:
    `d2d6aa7ba13dc3e762fe0abdc0790d2124f55945`
  - Closed the SLM-CPU-088 tracker state after #459.
  - Claim boundary: tracker closeout only; no additional runtime, release, or
    promotion claim.
- #465 `feat(cuda): add qwen3 short-decode-32 run 02 receipt`
  - Squash commit:
    `ecd533bdc816319b970e63edf4cf650f62c58419`
  - Added the second current-source Qwen3 `short_decode_32` strict CUDA
    receipt for CUDA-MODEL-017.
  - Claim boundary: one source receipt only; no aggregate completion, speedup,
    full residency, dense GGUF readiness, logits transfer reduction, Qwen2.5
    inheritance, BitNet packed I2_S/QK256 proof, or runtime/kernel/tokenizer/
    loader/server/routing claim.
- #463 `diag(a770): add A770-035 final block source frontier`
  - Squash commit:
    `9933d6eed73e111b3945daad1a9872c607dea20f`
  - Added A770-035 final-block source-frontier diagnostics, reports, and
    receipts.
  - Follow-up note: review found the landed `feed_forward_output` source
    fingerprint recorded the pre-FFN normalized input instead of the actual FFN
    branch output. #467 corrects that diagnostic source before this checkpoint
    treats `main` as the next promotion candidate.
  - Claim boundary: diagnostic source-frontier evidence only; no A770 support,
    quality, speed, parity, server readiness, route promotion, or completion
    claim.
- #466 `docs(a770): close A770-035`
  - Squash commit:
    `189207975c374b4b15fc05c8a7c4f3767ad36314`
  - Closed the A770-035 tracker state after #463.
  - Operational note: this tracker closeout merged before #467. #467 landed
    afterward as the required runtime diagnostic correction.
  - Claim boundary: tracker closeout only; no runtime math, CPU/A770 parity,
    answer readiness, broad quality, residency, speed, trusted partial
    acceleration, full inference, or BitNet QK256/I2_S behavior claim.
- #467 `fix(a770): record final block ffn output source`
  - Squash commit and current queue-stabilized promotion candidate SHA:
    `b38ba412c2928da42c2537dd41c015326e1e1768`
  - Corrected final-block source diagnostics so `feed_forward_output` records
    the actual FFN branch output before the residual add.
  - Claim boundary: narrow diagnostic correctness fix only; no support,
    quality, parity, speed, residency, route promotion, release, or source
    promotion claim.

## Closed Unmerged

- None in this checkpoint window.

## Open Queue After #467

- No open PRs in `EffortlessMetrics/bitnet-rs-swarm` at the post-#467
  checkpoint.

## Source Delta

- Current source ref:
  `source/main` = `ef6eec8a6f95a54138fd69617235347944d2caae`
- Current swarm ref:
  `origin/main` = `b38ba412c2928da42c2537dd41c015326e1e1768`
- `source/main` is reachable from `origin/main`.
- `rtk git rev-list --count source/main ^origin/main` returned `0`.
- `rtk git rev-list --count origin/main ^source/main` returned `85`.
- This handoff does not start a swarm-to-source promotion. The recorded SHA is
  a promotion candidate only after the source-promotion operator chooses the
  batch boundary, prepares the promotion packet, and verifies source-owned
  release surfaces.

## Evidence

- Open swarm PR queue at the post-#467 checkpoint:
  `rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --limit 100 --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt,isDraft`
  returned `[]`.
- #467 auto-merged by squash after `BitNet Rust Small on GitHub Hosted` and
  the normalized `BitNet Rust Small Result` completed with `SUCCESS`.
- `rtk git log --oneline --decorate -8 origin/main` showed #467 as the current
  `origin/main` tip after #466, #463, #465, #464, and #459.
- `rtk grep "record_final_block_source_tensors" crates/bitnet-transformer/src/lib.rs`
  showed the final-block recorder call on current `main`.
- `rtk grep "feed_forward_output_for_source" crates/bitnet-transformer/src/lib.rs`
  returned no matches on current `main`.
- `rtk git fetch origin --prune` and `rtk git fetch source --prune` both
  completed before recording the source delta.

## Validation

- #467 local proof before opening:
  - `rtk rustfmt --edition 2024 --check crates/bitnet-transformer/src/lib.rs`
  - `rtk git diff --check`
  - `rtk powershell -Command '$env:CARGO_TARGET_DIR="D:\codex-targets\bitnet-rs-swarm-a770035-fix"; rtk cargo check --locked -p bitnet-transformer --no-default-features --features cpu'`
  - `rtk powershell -Command '$env:CARGO_TARGET_DIR="D:\codex-targets\bitnet-rs-swarm-a770035-fix"; rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli final_block_source -- --nocapture'`
- #467 remote proof:
  - `BitNet Rust Small on GitHub Hosted`: `SUCCESS`
  - `BitNet Rust Small Result`: `SUCCESS`
- Validation gap: no source promotion proof was run. This checkpoint records a
  swarm candidate SHA only.

## Remaining Work

- Do not start a swarm-to-source promotion from this handoff alone. Use
  `b38ba412c2928da42c2537dd41c015326e1e1768` as the current
  queue-stabilized promotion candidate SHA only after the source-promotion
  operator chooses the batch boundary and verifies source-owned release
  surfaces.
- Continue processing new swarm PRs as they arrive: classify lane, inspect
  exact diff, verify claim boundaries, run focused proof, and squash-merge
  normal swarm PRs only when green.
- Keep source-history repair, source-to-swarm sync, and swarm-to-source
  promotion ancestry-preserving. Do not squash those repository-boundary PRs.
- Keep generated dashboards, campaign `active.toml`, workflows, `Cargo.lock`,
  root policy files, and authority docs under short merge-window discipline.

## Blockers

- None for the current handoff checkpoint.
- Source promotion has not started and remains a separate, ancestry-preserving
  release/source-owner operation.

## Claim Boundary

This handoff records operator state only. It does not claim A770 support, CUDA
support, Apple M4 support, Lunar Lake readiness, SLM CPU release readiness,
quality, speed, server readiness, selected attention, resident KV cache, full
residency, reference parity, source release readiness, source-to-swarm sync
completion, or swarm-to-source promotion completion.

## Next Operator Commands

```powershell
rtk gh pr list --repo EffortlessMetrics/bitnet-rs-swarm --state open --json number,title,headRefName,mergeStateStatus,autoMergeRequest,updatedAt,statusCheckRollup
rtk git fetch origin --prune
rtk git fetch source --prune
rtk git rev-parse origin/main
rtk git rev-parse source/main
rtk git merge-base --is-ancestor source/main origin/main
rtk git rev-list --count source/main ^origin/main
rtk git rev-list --count origin/main ^source/main
rtk git log --oneline -12 origin/main
```
