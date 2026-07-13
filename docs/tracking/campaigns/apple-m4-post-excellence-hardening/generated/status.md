<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 post excellence hardening Campaign Status

- Campaign: `apple-m4-post-excellence-hardening`
- State: `active`
- Objective: Keep the completed M4 Apple Silicon inference surface moving forward by hardening real user paths: dense local server behavior, timeout/cancellation, response conformance, BitNet repaired-quality analysis, benchmark variance, operator classes, and release-envelope refresh without reopening apple-m4-inference-excellence.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-HARDEN-001 | merged | #1087 | `codex/apple-m4/M4-HARDEN-001-seed-post-excellence-hardening` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Seed the apple-m4-post-excellence-hardening campaign and docs/tracking queue only, linking to the completed BitNet-rs apple-m4-inference-excellence completion audit, recording that the old lane is complete, and defining the dense SLM server hardening, BitNet repair, benchmark variance, and operator-envelope queues without runtime changes. |
| M4-HARDEN-002 | merged | #1091 | `codex/apple-m4/M4-HARDEN-002-dense-server-timeout-cancel` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Enforce dense local-server request timeout, prove streaming cancellation, emit partial-generation receipts, record stop reason, prove later requests still work, keep health/ready cheap, preserve fallback_used=false, and do not enable BitNet serve. |
| M4-HARDEN-003 | merged | #1580 | `codex/apple-m4/M4-HARDEN-003-dense-server-response-conformance` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Lock dense local-server non-streaming response shape, streaming event shape, and receipt export path; ensure bad model IDs fail cleanly, unsupported BitNet serve fails cleanly, and model/backend/fallback fields are visible in receipts. |
| M4-HARDEN-004 | merged | #1587 | `codex/apple-m4/M4-HARDEN-004-bitnet-repaired-250-analysis` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Classify the repaired 250-case BitNet regressions by task family, isolate scorer/template/runtime/model causes, compare against baseline, recommend a repair path, do not expand to 500, and do not enable chat or serve. |
| M4-HARDEN-005 | merged | #1603 | `codex/apple-m4/M4-HARDEN-005-bitnet-repaired-subset-rerun` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Rerun only the failing or repaired BitNet task families, prove improvement or document non-improvement, preserve model/tokenizer/prompt/backend identity, and keep dense and BitNet evidence separate. |
| M4-HARDEN-006 | in_progress | TBD | `codex/apple-m4/M4-HARDEN-006-benchmark-operator-envelope` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Record benchmark repeat count, p50/p90/p99/min/max, memory drift, timing variance, outlier handling, and advisory-vs-failure thresholds; then classify operator envelopes as interactive, advisory, batch, diagnostic, or unsupported. |

## Hard Constraints

- Use ci/hardware/apple-m4-mac-mini/2026-05-22/m4-inference-excellence-completion-audit.json as the completed baseline evidence.
- Do not reopen apple-m4-inference-excellence unless a real regression proves the baseline wrong.
- Do not rerun the whole M4 excellence proof set for this lane.
- Do not use dense SLM server evidence as BitNet chat or BitNet serve enablement evidence.
- Do not enable BitNet chat or BitNet serve in the dense server items.
- Do not expand the repaired BitNet 250-case corpus to 500 before the regression analysis item completes.
- Do not mix dense SLM and BitNet evidence, receipts, or claims.
- Do not claim full apple-m4-metal inference.
- Do not claim QK256-on-Apple support.
- Do not claim Neural Engine or MPSGraph model inference.
- Do not use MacBook evidence as M4 Mac mini runtime proof.
- Do not claim broad Apple Silicon support, broad model quality, or speedup.
- Do not commit model binaries.
