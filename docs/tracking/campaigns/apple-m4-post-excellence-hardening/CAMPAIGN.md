# Apple M4 Post Excellence Hardening Campaign

Campaign ID: `apple-m4-post-excellence-hardening`

Status: active

## Objective

Keep the completed M4 Apple Silicon inference surface moving forward by
hardening the real user paths: dense local server behavior,
timeout/cancellation, response conformance, BitNet repaired-quality analysis,
benchmark variance, operator classes, and release-envelope refresh.

## Source Of Truth Baseline

The completed baseline is the `apple-m4-inference-excellence` closeout imported
from `EffortlessMetrics/BitNet-rs`:

- `ci/hardware/apple-m4-mac-mini/2026-05-22/m4-inference-excellence-completion-audit.json`
- `docs/slm/apple-m4-inference-excellence.md`
- `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml`
- `docs/tracking/campaigns/apple-m4-inference-excellence/generated/status.md`

That audit records that all `apple-m4-inference-excellence` items are merged,
`campaign next apple-m4-inference-excellence` reports no next item, and the
completed lane remains the baseline source of truth. This campaign does not
reopen that lane and does not rerun the whole excellence proof set unless a
specific regression requires it.

The closeout also keeps the service boundary explicit: BitNet chat and serve
route contracts exist, but remain disabled unless the matching ready-gate
receipts are supplied by a future scoped item. Dense local-server evidence must
not be used as BitNet chat or BitNet serve enablement evidence.

## Hardening Queues

### Dense SLM server hardening

- Enforce request timeout for dense local-server generation.
- Make streaming cancellation observable and receipt-backed.
- Emit partial-generation receipts with stop reason.
- Prove later requests still work after timeout or cancellation.
- Keep health and ready cheap.
- Preserve `fallback_used=false` and exact model/backend identity.

### Dense SLM response conformance

- Lock non-streaming response shape.
- Lock streaming event shape.
- Lock receipt export path.
- Fail bad model IDs cleanly.
- Fail unsupported BitNet serve requests cleanly.
- Keep model, backend, and fallback visible in receipts.

### BitNet repair analysis

- Analyze the repaired 250-case BitNet regression before expanding.
- Classify regressions by task family.
- Isolate scorer, template, runtime, and model causes.
- Compare against the baseline evidence.
- Recommend a repair path without expanding to 500 cases.

### Benchmark variance

- Record repeat count, p50, p90, p99, min, max, memory drift, timing variance,
  and outlier handling.
- Separate advisory thresholds from failure thresholds.

### Operator envelope

- Refresh operator classes from receipt-backed evidence only:
  `interactive`, `advisory`, `batch`, `diagnostic`, and `unsupported`.
- Keep dense SLM, BitNet, server, benchmark, and release-envelope evidence
  separate.

## End State

- Dense local-server timeout and cancellation behavior is enforced, tested, and
  receipt-backed without enabling BitNet serve.
- Dense local-server response, streaming event, and receipt export shapes are
  locked for supported dense SLM routes and fail closed for unsupported routes.
- The repaired BitNet 250-case regression is explained before any corpus
  expansion.
- A repaired subset rerun either proves improvement for the failing families or
  records non-improvement with unchanged identity.
- Benchmark variance reports include repeat counts, percentile summaries,
  memory drift, timing variance, outlier policy, and advisory/failure
  thresholds.
- The operator envelope maps current evidence into `interactive`, `advisory`,
  `batch`, `diagnostic`, and `unsupported` classes without broad Apple Silicon,
  quality, or speed claims.

## Hard Constraints

- Do not reopen `apple-m4-inference-excellence` unless a real regression proves
  the baseline wrong.
- Do not rerun the whole M4 excellence proof set for this lane.
- Do not use dense SLM server evidence as BitNet chat or BitNet serve
  enablement evidence.
- Do not enable BitNet chat or BitNet serve in the dense server items.
- Do not expand the repaired BitNet 250-case corpus to 500 before the
  regression analysis item completes.
- Do not mix dense SLM and BitNet evidence, receipts, or claims.
- Do not claim full `apple-m4-metal` inference.
- Do not claim QK256-on-Apple support.
- Do not claim Neural Engine or MPSGraph model inference.
- Do not use MacBook evidence as M4 Mac mini runtime proof.
- Do not claim broad Apple Silicon support, broad model quality, or speedup.
- Do not commit model binaries.

## Work Items

The completed `apple-m4-inference-excellence` campaign already owns the
`M4-SERVE-EX-*` item IDs. This follow-on lane uses `M4-HARDEN-*` item IDs for
the next server-hardening work so the tracker does not duplicate or reopen
completed items.

| Work item | Status | Notes |
|---|---|---|
| M4-HARDEN-001 | merged | Seed this campaign and docs/tracking queue only. No runtime changes. |
| M4-HARDEN-002 | merged | Dense server timeout and cancellation with partial-generation receipts. No BitNet serve. |
| M4-HARDEN-003 | merged | Dense server response conformance, streaming events, receipt export, and clean failure paths. |
| M4-HARDEN-004 | merged | BitNet repaired-250 regression analysis before corpus expansion. |
| M4-HARDEN-005 | merged | Rerun only failing or repaired BitNet families and preserve identity. |
| M4-HARDEN-006 | in_progress | Benchmark variance, operator classes, and operator-envelope refresh. |

## Review Policy

Each PR owns one item. Runtime PRs must link to the exact work item, record the
receipt paths they create or consume, and keep dense SLM, BitNet, benchmark,
server, and operator-envelope claims separate. Generic required PR CI remains
model-free unless the item is a small parser, schema, fixture, docs, or
receipt-validation change.
