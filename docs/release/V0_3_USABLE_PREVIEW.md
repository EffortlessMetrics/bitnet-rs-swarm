# v0.3 Usable Preview Release Contract

Status: Active release-lane contract
Owner: BitNet-rs maintainers
Created: 2026-06-05
Linked proposal: n/a
Linked specs:

- `docs/specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md`
- `docs/specs/BITNET-SPEC-SUPPORT-BUNDLE.md`

Linked ADRs:

- `docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md`
- `docs/adr/BITNET-ADR-0011-lean-opt-in-github-hosted-fallback.md`

Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines the v0.3 usable-preview release bar; does not
promote a support tier by itself.
Policy impact: No policy exception.

This contract defines what BitNet-rs must prove before a public source
promotion can describe the project as a small, honest, installable local
inference preview. It is the release lane's governing contract until a later
source-owned release packet replaces it.

The contract does not release, publish, tag, or promote any runtime claim. It
sets the boundary for later PRs and source-promotion packets.

## Release Goal

The v0.3 usable preview is releasable when a user can:

1. build or install the `bitnet` CLI;
2. see supported, candidate, diagnostic, and unsupported model/device rows;
3. fetch and verify a supported model artifact;
4. run one local answer path for an exact supported model/device route;
5. inspect the receipt for that run;
6. generate an issue-safe support bundle;
7. understand what the receipt proves and what it does not prove.

This is not a production-readiness claim. It is a bounded preview claim for
exact model, artifact, tokenizer, prompt, backend, device, and receipt rows.

## User Path

The release must make this path obvious:

```bash
bitnet model status
bitnet model fetch <supported-model>
bitnet model verify <supported-model>
bitnet ask --model <supported-model> --device <supported-device> "What is 2+2?"
bitnet receipts explain --latest
bitnet support bundle --latest
```

The output must make these fields visible where the command has authority:

```text
model status: supported preview / candidate / diagnostic / unsupported
device status: supported preview / candidate / diagnostic / unsupported
artifact identity: model id, source, revision, file, size, sha256
tokenizer authority: source, path, sha256 when available
prompt authority: template or prompt contract used by the supported route
requested backend and selected backend
fallback_used
quality gate state
receipt path
speedup_claim
server_ready and server_ready_scope
claim boundary
next proof or blocker
```

Status commands may summarize support without executing inference. Receipts
prove what actually ran.

## Source Of Truth

| Surface | Authority |
| --- | --- |
| Support summary | `docs/status/SUPPORT_MATRIX.md` |
| Model coverage | `ci/model-artifacts/model-coverage-matrix.toml` and `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md` |
| BitNet route posture | `docs/status/BITNET_CAPABILITY_MATRIX.md` |
| CUDA route posture | `docs/status/CUDA_CAPABILITY_MATRIX.md` |
| Apple route posture | `docs/status/APPLE_CAPABILITY_MATRIX.md` |
| Hardware identity | `docs/hardware/HARDWARE_MATRIX.md` |
| Answer readiness | `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md` |
| Receipt explanation | `docs/specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md` |
| Support bundle | `docs/specs/BITNET-SPEC-SUPPORT-BUNDLE.md` |
| CI routing and cost policy | `docs/ci/cost-and-verification-policy.md` |
| User quickstart | `docs/quickstart.md` |

If a README, quickstart, or CLI help string disagrees with a receipt, model
coverage row, hardware matrix, or status page, the narrower claim wins.

## Minimum Releasable Rows

These rows define the target product surface. They are releasable only when the
linked authority proves the row at release time.

| Row | Release posture | Required proof | Not allowed to claim |
| --- | --- | --- | --- |
| Official Microsoft BitNet 2B I2_S/QK256 CPU | Supported preview when the model coverage row, CPU answer receipt, tokenizer/prompt authority, and receipt explanation all pass. | `bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s`; an accepted CPU answer receipt; `bitnet receipts explain <receipt>`; model coverage row `bitnet_official_2b_i2s_qk256`. | TL1/TL2, dense SLM support, CUDA speedup, server readiness, or full residency. |
| Official Microsoft BitNet 2B I2_S/QK256 CUDA | Supported preview only for exact CUDA rows whose receipts match `nvidia-rtx-5070-ti-cuda` and `bitnet_qk256_cuda`. | CUDA answer receipt with requested/selected backend, route, fallback, quality, and claim-boundary fields; `docs/status/CUDA_CAPABILITY_MATRIX.md`. | Generic CUDA, generic GPU, WGPU/Vulkan, A770, speedup, full residency, or broad server readiness. |
| Qwen2.5 0.5B Q8_0 dense SLM | Supported preview for exact rows named by the support and device matrices. | `bitnet model verify qwen2.5-0.5b-instruct-q8_0`; exact dense route receipts; matrix row proving the selected device. | BitNet QK256 proof, 1-bit support, broad dense model support, or speedup. |
| Qwen3 0.6B Q8_0 dense SLM | Supported preview only for bounded CUDA CLI paths if the support matrix keeps the row promoted. | `bitnet model verify qwen3-0.6b-instruct-q8_0`; exact CUDA answer receipt family. | Qwen2.5 inheritance, BitNet proof, broad dense support, speedup, server readiness, or full residency. |
| Apple M4 CPU/NEON dense SLM rows | Supported preview only for exact Apple rows listed in `docs/status/APPLE_CAPABILITY_MATRIX.md`. | Apple model verify, `bitnet mac ask`, `bitnet mac validate`, and accepted Apple CPU/NEON receipts for the exact model. | Metal, MPSGraph, Neural Engine, MacBook, broad Apple Silicon, BitNet QK256, or speed. |

## Candidate Or Diagnostic Rows

These surfaces may appear in status output, but they must not be presented as a
supported local-answer path unless a later exact proof row promotes them:

- SmolLM2 360M and other structurally valid model candidates;
- TL1, TL2, and BF16/GPU-int2 BitNet routes;
- A770, ROCm, WGPU, Vulkan, OpenVINO, Metal full inference, MPSGraph, and
  Neural Engine routes;
- server inference, except for an exact model/device/profile row with an
  accepted server receipt;
- speedup, except for an exact benchmark-qualified receipt and support row.

Diagnostic receipts remain useful support evidence. They are not product
support claims.

## Server And Speed Posture

Server readiness is false by default. A server route can enter the usable
preview only when an exact model, device, backend, endpoint, streaming profile,
receipt, and support-matrix row promote that profile. CLI answer readiness does
not imply server readiness, and dense SLM server proof does not imply BitNet
QK256 server proof.

Speedup is false by default. CUDA, Apple CPU/NEON, Metal, OpenVINO, WGPU,
Vulkan, or any other accelerator execution is not a speedup claim unless an
exact benchmark-qualified receipt and support row promote speed for the same
model, device, route, and runtime profile.

## Required CLI Behavior

The release CLI must fail closed for supported-preview commands:

- `bitnet model status` shows supported, candidate, diagnostic, unsupported,
  next proof, speed state, and server state without requiring the hardware to
  be present.
- `bitnet model fetch` reports cache location, source identity, expected
  artifact identity, and the distinction between artifact verification and
  answer readiness.
- `bitnet model verify` reports actual artifact identity, tokenizer and prompt
  authority when available, and the next proof needed for answer readiness.
- `bitnet ask` writes a receipt by default, prints the receipt path, summarizes
  fallback, backend, quality gate, and claim boundary, and refuses hidden
  fallback under strict routes.
- `bitnet receipts explain` maps a receipt back to model coverage, tier, route,
  backend, fallback, speed state, server state, claim boundary, and warnings.
- `bitnet support bundle --latest` produces a compact, issue-safe bundle with
  version, OS/CPU/GPU, model status, receipt explanation, last receipt, feature
  flags, relevant logs, and a machine-readable summary.

## Maintainer Proof Commands

Before a source promotion or preview tag, maintainers must record the exact
commands and outputs used. The minimum static proof set is:

```powershell
git diff --check
cargo run --locked -p xtask --no-default-features -- release-ready --profile usable-preview
cargo run --locked -p xtask --no-default-features -- check-model-coverage
npx --yes markdownlint-cli2@0.18.1 --config .markdownlint.jsonc docs/release/V0_3_USABLE_PREVIEW.md docs/status/SUPPORT_MATRIX.md docs/status/CUDA_CAPABILITY_MATRIX.md docs/status/APPLE_CAPABILITY_MATRIX.md
```

The `release-ready --profile usable-preview` guard must block public release
readiness while any critical proof remains unknown, including the exact local
answer receipt, fallback state, receipt explanation, speed posture, server
posture, support matrices, quickstart/README claim language, release notes, and
known limitations.

The minimum product proof set is exact-profile and may be satisfied by fresh
local execution or by committed accepted receipts:

```bash
bitnet model status --format json
bitnet model status --device <supported-device> --format json
bitnet model fetch <supported-model>
bitnet model verify <supported-model>
bitnet ask --model <supported-model> --device <supported-device> "What is 2+2?"
bitnet receipts explain --latest --format json
bitnet support bundle --latest --device <supported-device> --format json
```

If a command cannot run, the release packet must record:

- command;
- reason it could not run;
- substitute receipt or proof, if any;
- whether the gap blocks the release.

## Release Packet Requirements

A source-promotion or preview-tag packet must include:

- included swarm PRs and commit range;
- excluded work and why it is not part of the release claim;
- touched product, docs, policy, status, receipt, and generated surfaces;
- model/device rows promoted by the release;
- exact proof commands and receipt paths;
- claim boundary for CPU, CUDA, Apple, server, speed, residency, and model
  quality;
- policy impact and any exceptions;
- rollback plan.

Source release, publish, signing, and tag actions remain source-repo owned.
Swarm PRs prepare promotable evidence; they do not release by themselves.

## Release Blockers

The release is blocked if any of these are true:

- no supported local-answer path has an accepted receipt;
- `bitnet model status` cannot show supported versus candidate versus
  diagnostic rows;
- fetch or verify output cannot identify the artifact and cache location;
- `bitnet ask` can silently fall back on a strict supported route;
- receipt explanation omits fallback, selected backend, quality gate, speed
  state, server state, or claim boundary;
- support bundle output is unsafe to attach to an issue;
- README, quickstart, support matrix, or release notes overclaim beyond the
  status pages and receipts;
- speedup, server readiness, GPU support, Apple support, or model-quality
  language lacks exact-profile proof;
- required model coverage, status, receipt, or CI checks are missing or
  contradicted.

## Out Of Scope

The usable preview does not require:

- production deployment support;
- broad OpenAI-compatible server support;
- generic GPU support;
- A770 diagnostic promotion;
- new unproven model families;
- full benchmark optimization;
- broad microcrate consolidation;
- source/swap governance changes;
- release, publish, signing, or tagging from the swarm repo.

Keep those lanes separate unless a later plan item explicitly makes them part
of the release packet.
