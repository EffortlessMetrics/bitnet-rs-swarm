# Apple M4 BitNet Repaired Subset Rerun

Status: H005 evidence artifact
Owner: Codex
Created: 2026-06-05
Linked proposal: n/a
Linked specs: `docs/tracking/campaigns/apple-m4-post-excellence-hardening/active.toml#M4-HARDEN-005`
Linked ADRs: n/a
Linked plan: `docs/tracking/campaigns/apple-m4-post-excellence-hardening/active.toml#M4-HARDEN-005`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: none

This note records the `M4-HARDEN-005` repaired-subset rerun. It reruns only the
H004 primary regressed BitNet task families plus `constrained_summary` as a
positive control. It does not expand to 500 cases, rerun the full repaired 250,
enable BitNet chat, enable BitNet serve, or use dense SLM evidence as BitNet
evidence.

## Inputs

- Subset corpus: `ci/quality/apple-m4-bitnet-eval-repaired-subset.yaml`
- Observed receipt: `ci/hardware/apple-m4-mac-mini/2026-06-05T112555Z/bitnet-eval-repaired-subset/answer-corpus.json`
- Matched baseline subset: `ci/hardware/apple-m4-mac-mini/2026-06-05T112555Z/bitnet-eval-repaired-subset/baseline-2026-05-18T1806Z-answer-corpus.json`
- Analysis receipt: `ci/hardware/apple-m4-mac-mini/2026-06-05T112555Z/bitnet-eval-repaired-subset/subset-rerun-vs-2026-05-18T1806Z.json`

The baseline subset is the 2026-05-18 repaired 250-case receipt filtered to the
same 175 H005 fixture IDs. The subset corpus copies the selected fixture IDs,
prompts, per-case token caps, gates, and scoring rules from
`ci/quality/apple-m4-bitnet-eval-seeded-corpus-250.yaml`.

## Result

| Measure | 2026-05-18 matched subset baseline | 2026-06-05 repaired subset rerun | Delta |
|---|---:|---:|---:|
| Quality passed | 142 / 175 | 136 / 175 | -6 |
| Quality failed | 33 / 175 | 39 / 175 | +6 |
| Scoring passed | 147 / 175 | 139 / 175 | -8 |
| Scoring failed | 28 / 175 | 36 / 175 | +8 |
| Timeouts | 0 | 0 | 0 |
| Generated tokens | n/a | 1751 | n/a |

The subset did not improve. It had 8 quality case improvements and 14 quality
case regressions. The scoring view had 6 case improvements and 14 case
regressions.

## Family Deltas

| Family | Cases | Quality delta | Scoring delta | H005 result |
|---|---:|---:|---:|---|
| `numeric_tolerance` | 35 | -3 | -3 | regressed |
| `ordering_sorting` | 20 | -1 | -2 | regressed |
| `required_forbidden_tokens` | 30 | -1 | -1 | regressed |
| `closed_label_classification` | 20 | -1 | -1 | regressed |
| `format_constrained_json` | 20 | -1 | -1 | regressed |
| `rewrite_normalized` | 20 | 0 | -1 | scoring regressed |
| `constrained_summary` | 30 | +1 | +1 | positive control improved |

The positive control improved, but the primary repaired families did not. The
net result keeps BitNet in repair mode.

## Identity And Validation

The observed receipt records the accepted Microsoft BitNet I2_S model SHA
`4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`, external
tokenizer SHA `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`,
`bitnetcpp-answer`, `apple-m4-cpu-neon`, `runtime_api = cpu`, and
`fallback_used = false`. The regression comparison matched context and emitted
26 advisory warnings, all tied to quality or task-family deltas.

`mac receipts-check` now accepts the H005 repaired-subset corpus name for this
BitNet eval receipt family. That checker change only admits the scoped subset
receipt shape; it does not enable any server route or broaden BitNet support.

## Decision

Do not rerun the full repaired 250 yet. Do not expand to 500. Do not benchmark
BitNet as a user-performance route yet. The next BitNet move should remain
repair-focused on the primary regressed families, with full repaired-250 rerun
allowed only after the subset improves under unchanged recorded identity.

## Claim Boundary

This artifact may claim only that the scoped BitNet repaired subset rerun did
not improve under the recorded M4 Mac mini BitNet identity. It must not claim
dense SLM evidence supports BitNet quality, BitNet chat, BitNet serve, broad
BitNet quality, broad speedup, full Metal, QK256-on-Apple, Neural Engine,
MPSGraph, MacBook evidence, broad Apple Silicon support, or committed model
binaries.
