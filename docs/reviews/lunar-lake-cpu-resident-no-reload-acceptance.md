# Lunar Lake CPU Resident No-Reload Acceptance Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Post-matrix refresh: 2026-06-01
Resident package refresh: 2026-06-02
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069), [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071), [#1122](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1122), [#1209](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1209), [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232), [#1277](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1277), [#1280](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1280), [#1281](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1281), [#1291](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1291), [#1311](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1311), [#1374](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1374)
Linked PRs: [#1085](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1085), [#1107](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1107), [#1132](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1132), [#1182](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1182), [#1194](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1194), [#1208](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1208), [#1233](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1233), [#1234](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1234), [#1255](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1255), [#1279](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1279), [#1290](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1290), [#1292](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1292), [#1319](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1319), [#1334](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1334)
Support-tier impact: no promotion; review-only CPU resident timing acceptance
Policy impact: no policy exception

## Recommendation

Treat #1069 as closed historical command-surface work and use #1232 as the
live resident Rust GGUF phase evidence contract. The current
`lunar-lake-cpu-slm-resident-session.json` is now the #1334 physical resident
package from the #1279 fixture and #1290/#1292 measured-field work, not the
older #1255-only diagnostic baseline.

The current package records that the resident Rust GGUF CPU prompt loop keeps
the model and tokenizer loaded once, then runs one first resident ask plus 32
warm asks after first across `regression_tiny`, `ask_short`, and `ask_normal`.
It now records prompt-token counts, prompt render, tokenize, prefill, first
token, decode, detokenize, quality gate, total response, generated-token ID
availability, deterministic text/IDs, fallback false, and before/after memory
lifecycle fields. It remains diagnostic instead of benchmark-ready because
profile `receipt_write_ms` and `telemetry_ms` are explicit `not_exposed` fields
under #1291/#1292. PR #1319 closed #1311 by allowing
`diagnostic_package_reviewable=true` while preserving strict
`resident_phase_qualified=false` and `benchmark_qualified=false`.
Issue #1374 now owns the follow-up aggregate/session overhead scope. It may
define scope-specific receipt-write or telemetry collection fields for future
diagnostic visibility, but those fields must not backfill profile phase timing
or relax qualification by implication.

Issue #1277 defines the source-command boundary for the resident physical
package.
The new resident-specific corpus lives at
`ci/quality/lunar-lake-resident-qwen25-cpu.yaml`. It keeps the same three
initial profile cases as the durability corpus but uses `repeat_runs=11`, so
`slm-warm-session --corpus ci/quality/lunar-lake-resident-qwen25-cpu.yaml`
produces 33 prompts: one first resident ask plus 32 warm asks after first. This
defines the resident source shape used by the current #1334 package. It does
not make the resident session benchmark-qualified while profile receipt-write
and telemetry timing remain contract-not-exposed fields.

Issue #1280 is closed by #1334. Issue #1281/#1290 added source-side prompt
render, quality gate, detokenization, and memory lifecycle measurements. The
tracking in #1291/#1292 kept receipt-write and telemetry phase timing explicit
`not_exposed` values. Issue #1311/#1319 added the diagnostic-reviewable status
so this package can be reviewed without becoming strict phase or benchmark
evidence.

Do not optimize CPU kernels, change route policy, tune thread defaults, promote
OpenVINO CPU, or claim CPU speedup from the current no-reload evidence.

## Current Evidence Snapshot

`lunar-lake-cpu-slm-resident-session.json` currently records:

| Field | Current value | Acceptance relevance |
| --- | --- | --- |
| Schema | `1.2.0` | Includes diagnostic-reviewable qualification from #1319 |
| Route | `dense_slm_default_cpu` | CPU correctness/fallback route |
| Backend | `cpu-rust` | Rust GGUF dense Qwen path |
| Runtime | `resident_cpu_rust_gguf` | Resident prompt loop context |
| Fallback | `false` | Required for strict CPU route evidence |
| Model loaded once | `true` | No model reload observed in source receipt |
| Tokenizer loaded once | `true` | No tokenizer reload observed in source receipt |
| Model load | 9892.535 ms | One-time load cost remains separate |
| Tokenizer load | 253.471 ms | One-time tokenizer load remains separate |
| Prompt count | 33 | One first resident ask plus 32 warm asks after first |
| Thread count | 1 | #1071 is closed by #1208; no thread-count tuning follows |
| Power scheme | `Balanced` | Measured context, not low-power evidence |
| AC/battery state | `AC` | Measured context, not battery-mode evidence |
| Memory before load | 20213760 bytes | Lifecycle field measured |
| Memory after load | 2792079360 bytes | Lifecycle field measured |
| Memory after first ask | 2802257920 bytes | Lifecycle field measured |
| Memory after warm loop | 2802737152 bytes | Lifecycle field measured |
| Prompt token counts | measured in profile summaries | Profile applicability field present |
| Prompt render timing | measured in profile summaries | Host setup field present |
| Quality gate timing | measured in profile summaries | Quality-check overhead field present |
| Receipt write timing | `not_exposed` | Receipt overhead gap remains |
| Telemetry timing | `not_exposed` | Measurement overhead gap remains |
| Measurement qualification | `resident_phase_blocked_for_measurement_qualification` | Strict phase qualification remains blocked by `receipt_write_ms` and `telemetry_ms` |
| Diagnostic package reviewable | `true` | #1319 allows review without strict qualification |
| Resident phase qualified | `false` | Phase package is diagnostic only |
| Benchmark qualified | `false` | No CPU speedup, OpenVINO CPU promotion, or route-policy claim |
| Warm asks after first resident ask | 32 observed, 30 required | Run-shape count is satisfied, but strict phase fields remain incomplete |

The useful current conclusion is:

```text
Removing per-prompt model/tokenizer reload does not make Rust GGUF CPU fast;
resident asks remain dominated by prefill, first-token, and decode timing.
```

The unsafe conclusion is:

```text
The CPU slow path is solved, CPU is optimized, or CPU timing can be compared
as benchmark-qualified against OpenVINO CPU.
```

## Current Command Surface

The current command surface preserves the resident no-reload diagnostic
boundary and now indexes the #1334 physical resident source package.

- `lunar-lake cpu-slm-resident-session` is a no-new-inference summarizer over
  `lunar-lake-cpu-slm-phase-attribution.json` and the physical
  `lunar-lake-resident-qwen25-cpu-warm-session.json` source receipt.
- It emits `artifact_kind=lunar_lake_cpu_slm_resident_session` with
  `proof_stage=resident_cpu_no_reload_timing_no_new_inference`.
- The builder validates model/tokenizer loaded-once fields, answer gates,
  fallback=false, determinism, and summary timing from the source receipt.
- Since #1255, the builder emits `measurement_qualification` and fails closed
  instead of letting no-reload diagnostics look benchmark-ready.
- Since #1319/#1334, the builder can mark the package
  `diagnostic_package_reviewable=true` when all diagnostic fields are measured
  and only contract-not-exposed profile fields remain.
- It does not make profile `receipt_write_ms` or `telemetry_ms` measured, and
  it does not expose utilization, frequency, or thermal timing unless a future
  source receipt records those fields.
- Future aggregate/session overhead fields should be named and qualified under
  #1374 before implementation; current profile fields remain `not_exposed`.
- It does not make resident CPU timing benchmark-equivalent to OpenVINO CPU;
  #1156 keeps comparison qualification blocked while model formats, timing
  scopes, prompt-render/tokenization accounting, or matched-profile evidence
  differ.

The next #1232 PR should not be another aggregate refresh from existing
receipts. It should be issue-shaped first, then either add a narrowly scoped
measurement source for a named remaining field or define a matched CPU
comparison plan.

The preferred source shape is now explicit:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  slm-warm-session `
  --model <qwen2.5-0.5b-instruct-q8_0.gguf> `
  --corpus ci/quality/lunar-lake-resident-qwen25-cpu.yaml `
  --strict-loader --strict-tokenizer `
  --fail-on-quality --require-determinism `
  --json-out ci/hardware/intel-258v/2026-05-08/lunar-lake-resident-qwen25-cpu-warm-session.json
```

The exact model path remains operator-local. The source receipt should be
summarized through `lunar-lake cpu-slm-resident-session` with
`--repeated-warm-session lunar-lake-resident-qwen25-cpu-warm-session.json`.
The current committed summary remains diagnostic-only because strict
qualification still requires fields that the current contract marks
`not_exposed`.

## Fresh Measurement Acceptance Rule

A strict-qualified #1232 resident CPU phase receipt must include a physical run
with all of these gates:

| Gate | Required evidence |
| --- | --- |
| Run shape | One model load, one tokenizer load, one first resident ask, then 30 additional warm asks |
| Profiles | `regression_tiny`, `ask_short`, and `ask_normal` before heavier profiles |
| Route identity | `dense_slm_default_cpu`, `cpu-rust`, `resident_cpu_rust_gguf`, fallback false |
| Load separation | Model load, model hashing, tokenizer load, and resident prompt-loop timings remain separate |
| Phase timing | Prompt render, tokenize, prefill, first token, decode, detokenize, quality gate, receipt write, telemetry, and total response |
| Summary stats | Min, mean, p95, and max per phase and profile |
| Prompt accounting | Prompt token count, generated token count, stop reason, and max-new-token config per sample |
| Memory lifecycle | Before load, after load, after first ask, and after warm loop |
| Buffer reuse | Prompt/token buffers, generated-token buffers, timing buffers, and stop policy reuse status |
| Telemetry context | Thread count, Windows power scheme, AC/battery state, thermal availability, utilization/frequency proxy when available |
| Gap handling | Unavailable fields use explicit `not_exposed`, `not_available`, or equivalent status with null values |
| Claim boundary | No CPU speedup, route-policy change, OpenVINO CPU promotion, power claim, or BitNet QK256/I2_S claim |

The receipt may use existing command surfaces, but it must not silently reuse
older source receipts while presenting the result as a fresh physical
measurement.

## Fail-Closed Conditions

| Condition | Required decision |
| --- | --- |
| Model or tokenizer reloads during the warm loop | Keep #1232 open and diagnose reload cause |
| First resident ask is not separated from warm repeats | Keep #1232 open; cold/warm accounting is ambiguous |
| Fewer than 30 additional warm asks are recorded for the fresh run | Candidate diagnostic only |
| Prompt render, quality gate, receipt write, or telemetry timing is missing | Keep the gap explicit; do not claim complete phase attribution |
| Missing phase values are encoded as zero | Reject the receipt shape |
| Fallback appears | Reject strict CPU route evidence |
| Prompt/generated token counts are missing | Block profile-applicability claims |
| Thread count, power scheme, and AC/battery state are missing | Keep platform-context gap open; #1071/#1208 already close only the completed thread/core matrix |
| OpenVINO CPU timing is compared without matched scope and model format boundary | No benchmark-qualified CPU speedup claim |
| Dense SLM CPU evidence is cited as BitNet QK256/I2_S proof | Reject the claim boundary |

## Route Consequences

### CPU Correctness And Fallback

Rust GGUF CPU remains the dense SLM correctness/fallback baseline while answer
gates pass and fallback remains false. #1232 measurement does not by itself
change route policy.

### CPU Optimization

CPU optimization remains blocked until #1232 or a successor issue identifies a
target and success metric. The #1334 package shows resident asks are dominated
by prefill, first-token, and decode timing, but it does not isolate
kernel-internal causes such as attention, MLP, dequantization, KV-cache writes,
sampling, or logits work. The #1071/#1208 thread/core matrix is already
complete and does not justify default thread tuning.

### OpenVINO CPU

OpenVINO CPU remains a separate candidate/control path. The #1122 CPU route
decision, landed by #1132, continues to block matched-format CPU speedup
language until model format, timing scope, prompt rendering, tokenization, and
profile coverage align.

### Low Power

No #1232 resident CPU evidence is battery-mode `low_power` evidence. POWER-006
still requires strict battery telemetry and a benchmark-qualified energy or
power proxy.

## Next Smallest PR

No additional qualification/status PR is currently needed after #1319/#1334.
The next #1232 follow-up should be issue-shaped before implementation. Good
next PR candidates are:

- scoped aggregate receipt-write or telemetry fields under #1374, only if that
  issue names their timing scope and qualification effect;
- a matched Rust GGUF CPU versus OpenVINO CPU comparison plan that names model
  format, prompt rendering, timing scope, and profile coverage boundaries;
- topology, affinity, utilization, frequency, or thermal evidence if those
  fields become available through a narrow source;
- a receipt or validator gap discovered by a future resident run.

Do not combine that PR with CPU optimization, OpenVINO CPU promotion, route
policy mutation, broad benchmark refresh, or generated-dashboard churn.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- fresh resident CPU measurement;
- route-policy mutation;
- CPU optimization;
- CPU speedup or benchmark-qualified OpenVINO CPU comparison;
- low-power or battery evidence;
- thread/core matrix evidence;
- native accelerator proof;
- BitNet QK256/I2_S behavior proof.

It only updates the acceptance boundary for the current resident Rust GGUF CPU
no-reload diagnostic package and the next #1232 research-shaped follow-up.
