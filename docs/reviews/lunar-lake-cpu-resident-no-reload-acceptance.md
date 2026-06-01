# Lunar Lake CPU Resident No-Reload Acceptance Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Post-matrix refresh: 2026-06-01
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069), [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071), [#1122](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1122), [#1209](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1209), [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232)
Linked PRs: [#1085](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1085), [#1107](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1107), [#1132](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1132), [#1182](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1182), [#1194](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1194), [#1208](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1208), [#1233](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1233), [#1234](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1234)
Support-tier impact: no promotion; review-only CPU resident timing acceptance
Policy impact: no policy exception

## Recommendation

Treat #1069 as closed historical command-surface work and use #1232 as the
live resident Rust GGUF phase evidence contract. The current
`lunar-lake-cpu-slm-resident-session.json` remains a useful no-reload
diagnostic baseline, not a fresh post-#1209 physical measurement package.

The current receipt proves that the existing resident Rust GGUF CPU prompt loop
can keep the model and tokenizer loaded once while recording per-profile timing
for `regression_tiny`, `ask_short`, and `ask_normal`. It does not satisfy the
fresh measurement target because it does not count one first resident ask plus
30 additional warm asks in a new run, and it still lacks direct prompt-render,
quality-gate, receipt-write, telemetry, and full memory lifecycle timing.

Do not optimize CPU kernels, change route policy, tune thread defaults, promote
OpenVINO CPU, or claim CPU speedup from the current no-reload evidence.

## Current Evidence Snapshot

`lunar-lake-cpu-slm-resident-session.json` currently records:

| Field | Current value | Acceptance relevance |
| --- | --- | --- |
| Schema | `1.1.0` | Includes explicit gap fields from #1085 |
| Route | `dense_slm_default_cpu` | CPU correctness/fallback route |
| Backend | `cpu-rust` | Rust GGUF dense Qwen path |
| Runtime | `resident_cpu_rust_gguf` | Resident prompt loop context |
| Fallback | `false` | Required for strict CPU route evidence |
| Model loaded once | `true` | No model reload observed in source receipt |
| Tokenizer loaded once | `true` | No tokenizer reload observed in source receipt |
| Model load | 13592.773 ms | One-time load cost remains separate |
| Tokenizer load | 568.949 ms | One-time tokenizer load remains separate |
| Prompt count | 30 | Existing warm-loop evidence count |
| Thread count | 1 | #1071 is closed by #1208; no thread-count tuning follows |
| Power scheme | `not_sampled_in_slm_cpu_warm_session` | Telemetry gap, not zero evidence |
| AC/battery state | `not_sampled_in_slm_cpu_warm_session` | Telemetry gap, not low-power evidence |
| Memory before load | `not_exposed` | Lifecycle gap remains |
| Memory after load | `not_exposed` | Lifecycle gap remains |
| Memory after first ask | `not_exposed` | Lifecycle gap remains |
| Memory after warm loop | 2801647616 bytes | Only after-loop memory sample is measured |
| Prompt token counts | `not_exposed` in profile summaries | Profile applicability gap remains |
| Prompt render timing | `not_exposed` | Host setup gap remains |
| Quality gate timing | `not_exposed` | Receipt overhead gap remains |
| Receipt write timing | `not_exposed` | Receipt overhead gap remains |
| Telemetry timing | `not_exposed` | Measurement overhead gap remains |

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
boundary, but it does not yet collect the physical resident phase run now
defined by #1232.

- `lunar-lake cpu-slm-resident-session` is currently a no-new-inference
  summarizer over `lunar-lake-cpu-slm-phase-attribution.json` and an existing
  `slm_cpu_warm_session` source receipt.
- It emits `artifact_kind=lunar_lake_cpu_slm_resident_session` with
  `proof_stage=resident_cpu_no_reload_timing_no_new_inference`.
- The builder validates model/tokenizer loaded-once fields, answer gates,
  fallback=false, determinism, and summary timing from the source receipt.
- It does not run a fresh physical resident CPU session, create one separated
  first resident ask followed by 30 additional warm asks, or measure prompt
  render, quality-gate, receipt-write, telemetry, full memory lifecycle,
  utilization, or frequency timing unless a future source receipt records
  those fields.
- It does not make resident CPU timing benchmark-equivalent to OpenVINO CPU;
  #1156 keeps comparison qualification blocked while model formats, timing
  scopes, prompt-render/tokenization accounting, or matched-profile evidence
  differ.

The next #1232 PR should add or use a measurement source that emits the
accepted fresh resident run shape below, then keep the existing
summarizer/validator path as the fail-closed guard for missing fields. It
should not be another aggregate refresh from existing receipts.

## Fresh Measurement Acceptance Rule

A future #1232 resident CPU phase receipt must include one fresh physical run
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

CPU optimization remains blocked until the fresh #1232 resident receipt
identifies a target and success metric. The #1071/#1208 thread/core matrix is
already complete and does not justify default thread tuning. The likely target
is still prefill, first token, or decode, but the current receipt does not
isolate kernel-internal causes such as attention, MLP, dequantization,
KV-cache writes, sampling, or logits work.

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

The next implementation PR should be a receipt/schema guard or measurement
command refresh that:

- distinguishes first resident ask from the 30 measured warm asks;
- records the missing prompt-render, quality-gate, receipt-write, and telemetry
  timers;
- captures the full memory lifecycle;
- preserves explicit `not_exposed` values for unavailable telemetry;
- keeps the closed #1071/#1208 thread/core matrix as review evidence, not a
  tuning unlock.

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

It only defines the acceptance boundary for the next resident Rust GGUF CPU
no-reload timing refresh.
