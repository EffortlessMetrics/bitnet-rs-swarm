# Apple M4 BitNet Repaired 250 Regression Analysis

Status: H004 analysis artifact
Owner: Codex
Created: 2026-06-05
Linked proposal: n/a
Linked specs: `docs/tracking/campaigns/apple-m4-post-excellence-hardening/active.toml#M4-HARDEN-004`
Linked ADRs: n/a
Linked plan: `docs/tracking/campaigns/apple-m4-post-excellence-hardening/active.toml#M4-HARDEN-004`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: none

This note classifies the latest repaired 250-case BitNet run for
`M4-HARDEN-004`. It is analysis only. It does not expand the corpus, enable
BitNet chat, enable BitNet serve, make a broad BitNet quality claim, or use
dense SLM evidence as BitNet evidence.

## Comparator

The strict comparator is the repaired baseline:

- Current: `ci/hardware/apple-m4-mac-mini/2026-05-20T0133Z/bitnet-eval-250-repaired/answer-corpus.json`
- Baseline: `ci/hardware/apple-m4-mac-mini/2026-05-18T1806Z/bitnet-eval-250-repaired/answer-corpus.json`
- Analysis receipt: `ci/hardware/apple-m4-mac-mini/2026-05-20T0133Z/bitnet-eval-250-repaired/regression-analysis-vs-2026-05-18T1806Z.json`

The original `2026-05-17T1903Z/bitnet-eval-250` run remains context, not a
strict regression comparator for this item. The repaired corpus records
`corpus_version = 2.1.0` and adds the repaired scoring-kind contract; strict
`mac regression` against the original run fails with
`scoring_summary.kinds mismatch`.

## Result

| Measure | 2026-05-18 repaired baseline | 2026-05-20 repaired run | Delta |
|---|---:|---:|---:|
| Quality passed | 205 / 250 | 199 / 250 | -6 |
| Quality failed | 45 / 250 | 51 / 250 | +6 |
| Scoring passed | 210 / 250 | 202 / 250 | -8 |
| Scoring failed | 40 / 250 | 48 / 250 | +8 |
| Changed cases | n/a | 26 | n/a |

The receipts match on the important identity fields: Microsoft BitNet I2_S model
SHA, external tokenizer SHA, `bitnetcpp-answer` prompt identity,
`apple-m4-cpu-neon`, `runtime_api = cpu`, `fallback_used = false`, deterministic
greedy generation, and zero timeouts.

## Family Classification

| Family | Quality delta | Scoring delta | Classification | Repair priority |
|---|---:|---:|---|---|
| `numeric_tolerance` | -3 | -3 | Largest regression | primary |
| `ordering_sorting` | -1 | -2 | Regressed | primary |
| `required_forbidden_tokens` | -1 | -1 | Regressed with high case churn | primary |
| `closed_label_classification` | -1 | -1 | Regressed | primary |
| `format_constrained_json` | -1 | -1 | Regressed | primary |
| `rewrite_normalized` | 0 | -1 | Scoring regressed without quality net change | primary |
| `constrained_summary` | +1 | +1 | Net improved with case churn | positive control |
| `fixed_table_qa` | 0 | 0 | Unchanged with case churn | monitor |
| `synthetic_extraction` | 0 | 0 | Unchanged with case churn | monitor |
| `arithmetic_exact` | 0 | 0 | Unchanged | monitor |

## Cause Isolation

Scorer is not the primary cause for repaired-vs-repaired regression. Both
repaired receipts use the same corpus id, corpus version, scoring schema,
scoring kinds, and deterministic fixture scoring. Most new failures are
`answer_content`, not a new scorer category.

Template drift is not indicated. Both repaired receipts use
`bitnetcpp-answer`, the same prompt identity SHA, and the same template SHA.

Runtime drift is not indicated by the receipts. Both repaired receipts use
`apple-m4-cpu-neon`, `runtime_api = cpu`, `fallback_used = false`, and zero
timeouts. The next subset rerun is still needed to separate model-output churn
from any hidden runtime nondeterminism.

The probable primary cause is model-output instability under the same identity,
with the strongest repair signal in numeric, ordering, required/forbidden token,
closed-label, JSON-format, and normalized rewrite cases.

## Repair Path

For `M4-HARDEN-005`, rerun only the primary regressed families:
`numeric_tolerance`, `ordering_sorting`, `required_forbidden_tokens`,
`closed_label_classification`, `format_constrained_json`, and
`rewrite_normalized`. Include `constrained_summary` as a positive control because
it improved while still showing case churn.

Preserve model SHA, tokenizer SHA, prompt identity, backend, runtime API, cache
state, generation parameters, and fixture ids. Compare the subset against the
2026-05-18 repaired baseline. Rerun the full repaired 250 only after the subset
improves. If the subset does not improve, keep BitNet in repair mode and do not
expand the corpus or benchmark BitNet as a user-performance path.

## Claim Boundary

This analysis may claim only that the repaired 250-case BitNet regression is
classified and has a recommended repair path. It must not claim 500-case
coverage, BitNet chat, BitNet serve, broad BitNet quality, broad speedup, full
Metal, QK256-on-Apple, Neural Engine, MPSGraph, MacBook evidence, broad Apple
Silicon support, or dense SLM evidence for BitNet behavior.
