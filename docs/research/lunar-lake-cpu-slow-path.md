# Lunar Lake CPU Slow-Path Research

Research issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1035

Decision issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1122

Closed post-matrix review issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1209

Live resident phase evidence issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232

Live matched CPU comparison issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1365

Closed resident source-shape issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1277 /
https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1279

Closed physical resident run issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1280 /
https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1334

Closed resident field-gap issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1281

Closed receipt-write/telemetry scope issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1291 /
https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1292

Closed resident reviewable/qualified contract issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1311 /
https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1319

Decision memo: [Lunar Lake CPU Route Decision Memo](../reviews/lunar-lake-cpu-route-decision.md)

Closed physical matrix follow-up: [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071) /
[#1208](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1208)

Closed source-receipt follow-up: [#1201](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1201) /
[#1207](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1207)

Closed command/receipt-builder follow-ups: [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069) /
[#1182](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1182),
[#1186](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186) /
[#1194](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1194)

Closed resident qualification follow-up: [#1255](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1255)

Research date: 2026-05-30
Post-matrix refresh: 2026-06-01
Post-source-run refresh: 2026-06-02
Post-field rerun refresh: 2026-06-02
Post-reviewability-contract refresh: 2026-06-02
Post-physical-package refresh: 2026-06-02

Repository: `EffortlessMetrics/bitnet-rs-swarm`

## Executive Summary

The current Lunar Lake CPU slow path is not explained by one cause.

The strongest evidence says:

- one-off Rust GGUF CPU asks pay a large fixed model-load cost;
- removing reload with a resident CPU session improves latency but does not
  make the route fast;
- after reload is removed, prefill and first-token work dominate short asks;
- decode throughput is also low enough to dominate longer outputs;
- tokenizer and prompt-template setup are visible but much smaller than model
  load, prefill, and decode;
- #1208 records the default / 1-thread / 4-thread / 8-thread physical
  thread/core matrix with no receipt gaps;
- the matrix does not show a useful thread-count win: default and `threads_1`
  both resolved to one effective thread, while `threads_4` and `threads_8`
  were slower for `regression_tiny`, `ask_short`, and `ask_normal`;
- the matrix still does not expose P-core/E-core placement, utilization,
  frequency, or thermal readings, so affinity and default-thread policy remain
  unproven.
- #1255 records the current resident CPU receipt as explicitly
  `resident_phase_blocked_for_measurement_qualification`, separating no-reload
  diagnostic readiness from benchmark-ready resident phase evidence.
- #1277 is closed by #1279. The source fixture
  `ci/quality/lunar-lake-resident-qwen25-cpu.yaml` keeps the
  `regression_tiny`, `ask_short`, and `ask_normal` cases and uses
  `repeat_runs=11`, producing 33 prompts and 32 warm asks after the first
  resident ask.
- #1280 is closed by #1334. The committed physical package records the pinned
  Qwen2.5 Q8_0 GGUF SHA, explicit `--device cpu`, 33 prompts / 32 warm asks,
  `fallback_used=false`, `model_loaded_once=true`, `tokenizer_loaded_once=true`,
  quality passing, and determinism passing.
- #1281 is closed by #1290, and the #1334 summary now shows
  `prompt_render_ms`, `quality_gate_ms`, `detokenize_ms`, and resident memory
  lifecycle samples measured in the committed resident summary. The remaining
  strict blockers are only `receipt_write_ms` and `telemetry_ms`.
- #1291 is closed by #1292. The accepted scope contract keeps profile
  `receipt_write_ms` and `telemetry_ms` explicit `not_exposed` fields in
  current resident summaries instead of backfilling them from aggregate/session
  observations unless a later contract defines the source, scope, and
  qualification rule.
- #1311 is closed by #1319. The accepted contract adds a separate
  diagnostic-reviewability status for packages whose only remaining blockers
  are the #1291/#1292 `receipt_write_ms` and `telemetry_ms`
  contract-not-exposed fields, while preserving strict
  `resident_phase_qualified=false` and `benchmark_qualified=false`.
- The #1334 resident summary now exposes that status:
  `diagnostic_package_reviewable=true`, `resident_phase_qualified=false`, and
  `benchmark_qualified=false`. The remaining resident blockers are only the
  per-profile `receipt_write_ms` and `telemetry_ms` fields.

The current route decision is:

- keep Rust GGUF CPU as the correctness, fallback, and comparison plate;
- treat OpenVINO CPU as a separate diagnostic candidate, not a drop-in
  replacement for the Rust GGUF CPU route;
- defer Rust GGUF CPU optimization until #1232 resident phase timing, #1365
  matched comparison evidence, or a later topology receipt identifies a single
  target.

Do not start a CPU runtime optimization PR from this research alone. The live
CPU resident planning issue is
[#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232),
with the closed
[#1280](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1280) /
[#1334](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1334)
physical package,
[#1291](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1291) /
[#1292](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1292)
receipt-field contract, and
the closed
[#1311](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1311) /
[#1319](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1319)
diagnostic-reviewability contract.
Together they define resident Rust GGUF phase evidence before any optimization,
or route-policy PR. The separate
[#1365](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1365)
issue now owns the matched Rust GGUF CPU versus OpenVINO CPU comparison package
and keeps benchmark qualification fail-closed while model format or timing
scope differ.

## Current CPU Route Context

`dense_slm_default_cpu` remains the promoted CPU reference route for strict
local regression and comparison work. It uses the Rust GGUF dense Qwen path:

- selected backend: `cpu-rust`;
- runtime API: `cpu`;
- selected runtime or kernel: `i2_s-avx2-reference` or
  `resident_cpu_rust_gguf`, depending on receipt;
- model: Qwen2.5-0.5B-Instruct Q8_0 GGUF;
- tokenizer source: GGUF metadata;
- fallback used: false in the cited receipts.

This research does not change that route. It only names the likely slow-path
causes and the measurement plan needed before a runtime change.

## Evidence Map

| Receipt | Scope | Key CPU findings |
| --- | --- | --- |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-phase-attribution.json` | Derived CPU phase attribution, no new inference | Cold one-off total response 27986.539 ms; cold load 14250.931 ms; tokenize 482.325 ms; prefill 9361.503 ms; first token 9726 ms; decode 3242.064 ms for 9 output tokens |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-resident-session.json` | Current #1334 resident Rust GGUF CPU prompt-loop summary, no model/tokenizer reload per prompt | Model loaded once; tokenizer loaded once; 33 prompts and 32 warm asks after first; ask_short mean total 4774.776 ms; ask_normal mean total 7271.792 ms; no fallback observed; quality and determinism passed; `diagnostic_package_reviewable=true`, `resident_phase_qualified=false`, and `benchmark_qualified=false`; remaining blockers are profile `receipt_write_ms` and `telemetry_ms` |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-resident-qwen25-cpu-warm-session.json` | #1334 physical resident Rust GGUF CPU source receipt from the committed fixture | Pinned Qwen2.5 Q8_0 GGUF SHA matched; `slm-warm-session --device cpu` produced 33 prompts and per-prompt receipts, selected backend `cpu-rust`, runtime `cpu`, fallback false, quality passing, deterministic generated IDs/text, and model/tokenizer loaded once |
| `ci/quality/lunar-lake-resident-qwen25-cpu.yaml` | Closed #1279 source fixture for the physical resident run, no inference by itself | 3 cases x `repeat_runs=11`, yielding 33 prompts / 32 warm asks after first. This fixed the source shape found in #1277 without changing route policy |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-profile-run.json` | Explicit Rust GGUF CPU heavy-profile timing | prefill_heavy total 1373681.117 ms for 2734 prompt tokens and 16 generated tokens; decode_heavy total 123115.592 ms for 67 prompt tokens and 512 generated tokens |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-runtime-comparison.json` | Refreshed Rust GGUF CPU versus OpenVINO CPU diagnostic comparison | Rust resident ask_short mean 11158.750 ms and ask_normal mean 16407.372 ms; OpenVINO CPU corpus-v2 now passes 14/14 with fallback false, but the receipt remains context-only because model format, timing scope, prompt-render, tokenization, and matched-profile gaps block benchmark qualification |
| `ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json` | Newer OpenVINO CPU/GPU/NPU corpus-v2 receipt | OpenVINO CPU resolved to `Intel(R) Core(TM) Ultra 7 258V`, constructed in 981.455 ms, ran 14/14 corpus-v2 cases with fallback false and direct generated token IDs |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-cpu-corpus-v2-diagnosis.json` | OpenVINO CPU diagnosis | OpenVINO CPU corpus-v2 diagnosis says 14 total, 14 passed, 0 failed, no fallback, direct generated token IDs available |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-thread-core-matrix.json` | Physical Rust GGUF CPU default / 1-thread / 4-thread / 8-thread resident matrix | `matrix_ready=true`, `gaps=[]`; default effective threads = 1; `threads_4` and `threads_8` are slower than default/1-thread across the three measured profiles; no speedup, tuning, route-policy, low-power, accelerator, or BitNet claim |

The OpenVINO CPU comparison evidence was refreshed against the newer
corpus-v2 run. Do not use the runtime-comparison receipt as a benchmark
speedup claim: Rust GGUF CPU uses Q8_0 GGUF and OpenVINO CPU uses INT4_SYM
OpenVINO IR, timing scopes still differ, OpenVINO tokenization/detokenization
metrics are not fully exposed, and several corpus-v2 profiles lack matched
Rust resident evidence. #1156 now enforces that boundary in the comparison
builder: benchmark qualification must remain false while model formats or
timing scopes differ.

## Top Likely Causes

### 1. Rust GGUF CPU Prefill And First-Token Cost

This is the strongest current root-cause candidate for resident asks.

The current #1334 resident summary removes per-prompt model and tokenizer
reload, measures the prompt loop from the physical #1279 fixture, and still
shows prefill, first-token, and decode as the material costs:

| Profile | Count | Mean total | Mean time to first token | Mean prefill | Mean decode | Mean tokenize | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ask_short` | 11 | 4774.776 ms | 3734.455 ms | 3588.079 ms | 1186.351 ms | 0.004 ms | 8 |
| `ask_normal` | 11 | 7271.792 ms | 3843.909 ms | 3697.454 ms | 3573.740 ms | 0.004 ms | 24 |
| `regression_tiny` | 11 | 5084.737 ms | 4064.818 ms | 3921.156 ms | 1163.243 ms | 0.004 ms | 8 |

The older #1086 runtime-comparison receipt still carries larger Rust resident
context totals: `ask_short` 11158.750 ms, `ask_normal` 16407.372 ms, and
`regression_tiny` 12314.790 ms. Keep those values as
runtime-comparison context only until a later #1365 matched comparison package
revises the comparison scope. Do not treat them as the current #1334 resident
qualification surface.

Warm phase receipts point the same way:

| Phase profile | Prompt tokens | Generated tokens | Prefill | Prefill per prompt token | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_512` | 541 | 1 | 63420.772 ms | 117.229 ms/token | 175.035 ms |
| `decode_128` | 37 | 128 | 3539.626 ms | 95.666 ms/token | 16931.925 ms |

The CPU route is therefore slow even after reload is removed. That points to
the Rust GGUF dense Qwen prefill/first-token path, not only CLI startup.

Uncertainty:

- receipts do not split attention, MLP, dequantization, KV-cache writes, RoPE,
  sampling, and logits work;
- thread placement, core class, and frequency are not recorded;
- the `first_token_ms` and `prefill_ms` fields overlap conceptually, so they
  should be interpreted as diagnostic phase evidence, not additive accounting.

### 2. Decode Throughput Is Low For Longer Outputs

Decode is not the main cost for a 9-token cold ask, but it becomes material
quickly.

Evidence:

- current #1334 resident `ask_short` generated 8 tokens at about
  6.745 tokens/s;
- current #1334 resident `ask_normal` generated 24 tokens at about
  6.716 tokens/s;
- warm `decode_128` generated 128 tokens at about 7.560 tokens/s;
- explicit `decode_heavy` generated 512 tokens with 111754.248 ms decode time,
  about 4.578 tokens/s.

This makes decode a first-class slow-path cause for `ask_normal`,
`decode_heavy`, and any route that needs sustained local conversation.

Uncertainty:

- the receipts do not isolate token sampling, logits transforms, and detokenize
  from model forward work;
- they do not show whether the decode loop reuses all buffers expected by the
  resident session;
- they do not record per-layer or per-kernel timing.

### 3. Cold Model Load Adds A Large Fixed Tax

Cold one-off routing is still strongly affected by model load.

`lunar-lake-cpu-slm-phase-attribution.json` records:

- total response: 27986.539 ms;
- cold load: 14250.931 ms;
- model-load share of total: 0.509;
- tokenize share of total: 0.017;
- decode share of total: 0.116;
- reported prefill share of total: 0.335.

The resident-session receipt records:

- model loaded once: true;
- tokenizer loaded once: true;
- resident model load: 9892.535 ms;
- model SHA-256: 240.837 ms;
- tokenizer load: 253.471 ms;
- prompt count: 33;
- warm asks after first resident ask: 32;
- no per-prompt model or tokenizer reload observed.

Removing reload improved total latency. The cold-to-resident total ratio was
5.861 for `ask_short`, 3.849 for `ask_normal`, and 5.504 for
`regression_tiny`.

That is enough to justify resident-session measurement, but not enough to say
reload is the full root cause.

## Less Likely Or Unproven Causes

### Tokenizer And Template Setup

Tokenizer work is measurable but not dominant in current receipts:

- cold one-off tokenize: 482.325 ms;
- current #1334 resident ask_short tokenize mean: 0.004 ms;
- current #1334 resident ask_normal tokenize mean: 0.004 ms;
- #1086 runtime-comparison context tokenize means: ask_short 466.206 ms,
  ask_normal 465.554 ms, and regression_tiny 476.496 ms;
- current #1334 resident tokenizer load: 253.471 ms as a one-time session
  cost, not a repeated per-prompt reload.

This is worth cleaning up after the larger prefill/decode and cold-load
questions, but it is not the top slow-path cause.

### Receipt Overhead

The resident receipt enabled per-prompt receipts and recorded reusable buffers,
but no receipt currently isolates:

- JSON serialization time;
- prompt sidecar write time;
- stdout or logging time;
- telemetry collection time.

PR #1290 now isolates quality-gate evaluation time. Receipt overhead may matter
for tight benchmark loops, but the existing latencies are too large for
receipt overhead to be the main explanation.

Issue #1280 is closed by #1334. The committed physical package records useful
resident means, and the strict summary now measures prompt rendering, quality
gates, detokenization, and memory lifecycle samples. The unresolved overhead
question is narrowed to `receipt_write_ms` and `telemetry_ms` only, and those
fields remain profile-level `not_exposed` blockers under the #1291/#1292
contract.

### Receipt Write And Telemetry Contract

Do not backfill per-profile resident phase metrics from aggregate/session
observations.

Accepted current contract from #1291/#1292:

- `receipt_write_ms` is not a per-prompt phase metric in current
  `slm-warm-session` receipts. A prompt receipt cannot truthfully include the
  elapsed time of writing itself unless the command writes a sidecar event or
  performs a two-pass receipt update. Until that design is accepted, resident
  summaries must keep profile `receipt_write_ms` as `not_exposed`.
- `telemetry_ms` is not a per-prompt profile metric in current CPU resident
  evidence. Power scheme, AC/battery state, process memory, and thermal
  availability are sampled as session context. A single aggregate probe must
  not be copied into every profile's phase timing. Until a per-prompt or
  profile-attributed telemetry probe exists, resident summaries must keep
  profile `telemetry_ms` as `not_exposed`.
- Aggregate timing fields may be added later, but they should be named by
  scope, for example `timing.aggregate_receipt_write_ms` or
  `telemetry.collection_ms`, and the resident summarizer should not use them to
  satisfy per-profile phase timing unless a spec explicitly says so.
- For the current Rust GGUF CPU resident-session scope, explicit
  `not_exposed` statuses for profile `receipt_write_ms` and `telemetry_ms`
  should remain measurement blockers for the strict #1232 contract, but they
  should not justify a behavior change or CPU optimization PR. A later
  contract may relax resident-phase qualification only if it records why
  aggregate/session overhead is outside the profile timing surface.
- #1311 is closed by #1319. Current summaries keep strict qualification false,
  but may expose a separate diagnostic-reviewable status when these two
  contractually unavailable fields are the only remaining blockers.
- `benchmark_qualified` remains false regardless of this contract unless the
  compared routes share benchmark-equivalent model format, prompt scope,
  timing scope, token visibility, and route identity.

### Thread And Core Behavior

Thread/core behavior is now measured for the basic default / 1 / 4 / 8 thread
matrix, but it is not an optimization target yet.

The 258V platform has 4 P-cores plus 4 low-power E-cores, and the roadmap says
power mode and thermal profile matter. #1208 records:

- default, `threads_1`, `threads_4`, and `threads_8` variants;
- requested and effective thread count;
- Windows `Balanced` power scheme and AC state;
- process affinity mask `0xff`;
- `not_exposed` statuses for affinity classification, thermal readings,
  utilization, and frequency/throttle proxy.

The matrix result does not support default thread tuning:

| Variant | Effective threads | `ask_normal` total mean | `ask_short` total mean | `regression_tiny` total mean |
| --- | ---: | ---: | ---: | ---: |
| `default` | 1 | 7472.066 ms | 4833.832 ms | 5056.348 ms |
| `threads_1` | 1 | 7434.133 ms | 4836.348 ms | 5174.859 ms |
| `threads_4` | 4 | 7771.171 ms | 5161.869 ms | 5479.246 ms |
| `threads_8` | 8 | 7895.855 ms | 5212.025 ms | 5578.867 ms |

Treat this as evidence against blind thread-count tuning on the current host
and corpus. It does not prove that affinity, P-core/E-core placement,
frequency behavior, or battery power mode are irrelevant, because those fields
remain unavailable or unmeasured in the current matrix.

## Rust GGUF CPU Versus OpenVINO CPU

The current comparison is useful but not action-ready.

Newer OpenVINO CPU corpus-v2 evidence:

- selected backend: `openvino-cpu`;
- runtime API: `openvino_genai`;
- resolved device: `Intel(R) Core(TM) Ultra 7 258V`;
- pipeline construct: 981.455 ms;
- fallback used: false;
- 14/14 corpus-v2 cases passed;
- direct generated token IDs available.

This table intentionally cites the #1086 runtime-comparison receipt, not the
newer #1334 resident summary. It remains useful for non-equivalent Rust GGUF
CPU versus OpenVINO CPU context, but it is not the current #1334 resident
qualification surface and is not benchmark-qualified.

OpenVINO CPU profile timing in that receipt is far lower than its Rust GGUF
CPU comparison context, for example:

| Profile | Rust GGUF CPU mean total | OpenVINO CPU mean generation wall | OpenVINO CPU cases |
| --- | ---: | ---: | ---: |
| `ask_short` | 11158.750 ms | 109.298 ms | 2 |
| `ask_normal` | 16407.372 ms | 232.792 ms | 3 |
| `regression_tiny` | 12314.790 ms | 250.786 ms | 4 |

Do not promote OpenVINO CPU from this table:

- the model formats differ: GGUF Q8_0 versus OpenVINO IR INT4_SYM;
- the timing scopes differ: Rust resident total versus OpenVINO generation
  wall plus pipeline construction elsewhere;
- OpenVINO GenAI tokenization and detokenization metrics report `-1.0` in the
  current corpus-v2 receipt, so host tokenizer/template setup is not fully
  split;
- matched Rust resident profile evidence is still missing for `decode_heavy`,
  `low_power`, `prefill_heavy`, `structured`, and `warm_resident`.

Use this evidence to justify a matched comparison receipt, not a route change.

## CPU Route Decision Memo

Decision date: 2026-05-31

This research resolves the immediate #1122 planning question as follows.

### Keep Rust GGUF CPU As Correctness/Fallback Plate

Keep `dense_slm_default_cpu` on the Rust GGUF CPU path for correctness,
fallback, regression, and route comparison. It is slow, but it is the local
route that preserves the current GGUF model, tokenizer source, prompt
template, receipt shape, and fallback-free regression context.

This is not a performance endorsement. It is a control-plane decision: the CPU
plate stays stable until another route has an exact-profile promotion package
or the CPU evidence identifies a narrow optimization target.

### Keep OpenVINO CPU Separate For Now

OpenVINO CPU should remain a separate route candidate and diagnostic reference.
The current OpenVINO CPU corpus-v2 evidence is strong enough to prove that the
OpenVINO CPU export can answer the bounded corpus with direct generated token
IDs and `fallback_used=false`. It is not strong enough to replace the Rust GGUF
CPU route because:

- Q8_0 GGUF and INT4_SYM OpenVINO IR are not model-format equivalent;
- Rust resident total time and OpenVINO generation wall time do not cover the
  same host setup, tokenizer, prompt-render, pipeline, and receipt phases;
- OpenVINO tokenization and detokenization phase metrics are still not fully
  exposed in the current corpus-v2 receipt;
- matched Rust resident evidence does not cover every OpenVINO profile.

Do not collapse OpenVINO CPU into the CPU default route until a later review
accepts a matched route/profile comparison despite those differences, or a
separate product decision explicitly introduces OpenVINO CPU as a distinct
promoted CPU profile.

### Do Not Optimize Blindly

The next code PR should not tune threads, kernels, tokenizer setup, or route
policy by intuition. The present evidence points at prefill, first-token,
decode, and thread/core behavior as the likely target set, but it does not yet
separate enough sub-phases to justify a durable runtime change.

Optimization becomes a good PR only after one of these is true:

- resident timing proves a repeated per-prompt sub-phase dominates after model
  and tokenizer load are excluded;
- a later topology or affinity receipt shows a stable, repeatable placement
  effect that can be guarded without hurting correctness or low-power evidence;
- matched Rust GGUF versus OpenVINO CPU comparison shows that the route decision
  is about model/runtime format, not a missing Rust instrumentation field.

## Fair CPU Benchmark Boundary

A fair CPU benchmark for route decisions must be profile-scoped and honest
about what differs. It must record:

- Rust GGUF model identity and OpenVINO IR model identity separately;
- quantization and format mismatch, including Q8_0 GGUF versus INT4_SYM
  OpenVINO IR;
- tokenizer source, chat template source, prompt-render policy, and generation
  config for each route;
- cold/warm mode, model or pipeline construction, tokenization, prefill,
  first-token, decode, detokenize, quality gate, receipt write, and total
  timing scope;
- prompt and generated token counts for the named profile;
- direct generated-token ID status, retokenized status, or unavailable status;
- selected backend/runtime/device and `fallback_used=false`;
- AC/battery state, Windows power scheme, and thermal availability;
- whether the comparison is route/profile evidence, model-format diagnostic
  evidence, or benchmark-qualified promotion input.

The benchmark must fail closed for promotion if it hides model-format mismatch,
uses different generation configs, lacks same-profile CPU evidence, lacks
fallback status, or treats a narrower OpenVINO timing scope as equivalent to
Rust total response timing.

## Measurement Plan

### CPU Phase Attribution Receipt

Add one small Rust GGUF CPU phase receipt that records a single process,
single loaded model, and fixed prompt set.

Required fields:

- `cold_warm_mode`
- `process_start_ms`
- `model_load_ms`
- `model_resolve_ms`
- `model_open_ms`
- `model_mmap_ms`
- `model_parse_metadata_ms`
- `model_tensor_index_ms`
- `model_weight_materialize_ms`
- `model_sha256_ms`
- `tokenizer_load_ms`
- `prompt_template_resolve_ms`
- `prompt_render_ms`
- `tokenize_ms`
- `prefill_ms`
- `first_token_ms`
- `decode_total_ms`
- `detokenize_ms`
- `quality_gate_ms`
- `receipt_write_ms`
- `telemetry_collect_ms`
- `total_response_ms`
- `timing_scope`
- `fallback_used=false`

This should target `regression_tiny`, `ask_short`, and `ask_normal` first. Do
not start with `prefill_heavy`.

Schema hardening should expose these fields on cold and warm profile samples.
Unavailable sub-phases must use `not_exposed` with null values; the receipt is
measurement infrastructure and is not a CPU performance fix or route promotion.

### Resident CPU Session Receipt

Refresh the resident session with a smaller, explicit before/after accounting
block:

- model loaded once;
- tokenizer loaded once;
- one cold first ask through the resident engine;
- 30 warm asks;
- per-profile mean/p95/max for tokenize, prefill, first token, decode,
  detokenize, quality gate, receipt write, and total;
- memory before load, after load, after first ask, and after warm loop;
- prompt/token buffers reused;
- generated-token buffers reused.

Acceptance: resident timing must keep model-load cost separate from per-prompt
cost and must state that resident proof does not remove cold-start cost.

The current committed resident receipt from #1334 now records the physical
resident phase package from the #1279 fixture. It fixes the old warm-count,
prompt-token, prompt-render, quality-gate, detokenization, and memory lifecycle
gaps while preserving the accepted status boundary:
`diagnostic_package_reviewable=true`, `resident_phase_qualified=false`, and
`benchmark_qualified=false`.

The #1334 measured means are diagnostic only:

| Profile | Count | Mean total | Mean TTFT | Mean prefill | Mean decode | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `regression_tiny` | 11 | 5084.737 ms | 4064.818 ms | 3921.156 ms | 1163.243 ms | 8 |
| `ask_short` | 11 | 4774.776 ms | 3734.455 ms | 3588.079 ms | 1186.351 ms | 8 |
| `ask_normal` | 11 | 7271.792 ms | 3843.909 ms | 3697.454 ms | 3573.740 ms | 24 |

The next narrow step is not another receipt-builder PR by default. The
closed #1291/#1292 contract already decides that current resident summaries
keep profile `receipt_write_ms` and `telemetry_ms` as explicit `not_exposed`
fields instead of backfilling aggregate/session observations. Keep
`resident_phase_qualified=false` unless the full #1232 contract is satisfied or
explicitly revised.

### Thread/Core Matrix Receipt

Detailed plan: `docs/research/lunar-lake-cpu-thread-core-matrix.md`.

Status: completed for the required default / 1-thread / 4-thread / 8-thread
matrix by #1208.

Run the same resident ask set across a small matrix:

| Variant | Purpose |
| --- | --- |
| current default | preserve baseline |
| `--threads 1` | isolate single-thread behavior |
| `--threads 4` | approximate P-core-only capacity without affinity proof |
| `--threads 8` | full logical 258V thread count |
| optional P-core affinity | test performance-core placement |
| optional E-core affinity | test low-power-core placement |

Required context, as captured or explicitly unavailable in #1208:

- Windows power scheme;
- battery or AC state;
- thermal availability;
- process affinity mask if set;
- effective environment thread vars;
- CPU utilization per logical processor if available;
- frequency/throttle proxy if available.

### Matched Rust GGUF CPU Versus OpenVINO CPU Receipt

Refresh the comparison with one receipt that names the mismatch instead of
hiding it:

- Rust GGUF Q8_0 model identity and OpenVINO IR INT4_SYM model identity;
- same corpus cases and generation config;
- same cold/warm mode;
- answer-gate result per profile;
- direct generated-token availability;
- pipeline/model construction separately from generation;
- tokenization and prompt-render status;
- explicit statement that route/profile comparison is not token-level engine
  parity and not a speedup claim unless benchmark qualification rules are met.

## Implementation Candidates

| Rank | Candidate | Expected signal | CI/runtime risk |
| ---: | --- | --- | --- |
| 1 | Apply the #1291/#1292 receipt-write and telemetry timing scope contract | High: preserves the accepted profile-versus-aggregate boundary and prevents backfilled strict fields | Low: docs/contract surface; no runtime behavior change |
| 2 | Use #1232 to decide the next resident phase-evidence follow-up after the committed #1280/#1334 package | High: keeps diagnostic-reviewable resident evidence from becoming an optimization or benchmark claim | Low: issue/research shaping before any new physical run |
| 3 | Refresh Rust GGUF CPU versus OpenVINO CPU comparison | Medium: clarifies whether OpenVINO CPU is a route candidate or only diagnostic context | Medium: OpenVINO hardware/software run, but docs/receipt only |
| 4 | Add scoped aggregate receipt-write or telemetry fields only if a later contract names their scope and qualification effect | Medium: could explain host overhead without corrupting per-profile phase timing | Low-medium: schema/receipt hardening if tightly scoped |
| 5 | Add a later affinity/topology receipt only if P-core/E-core placement can be exposed accurately | Medium: may explain placement behavior the current matrix could not expose | Medium: requires Windows affinity and scheduler care |
| 6 | Optimize tokenizer/template setup | Low-medium: visible hundreds of milliseconds, but not dominant | Low-medium: local code change risk depends on tokenizer ownership |
| 7 | Change prefill/decode kernels or route policy | Potentially high, but evidence not yet precise enough | High: broad runtime and CI churn risk |

## Current Next Steps

`CPU-SLM-PERF-001` was opened as
[#1045](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1045) and
is now closed. Treat its receipt/schema hardening as the completed first step,
not as the next action.

The live CPU slow-path follow-ups and guard status are:

1. [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069)
   is closed by [#1182](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1182)
   as a no-new-inference `lunar-lake cpu-slm-resident-session`
   command-surface review. It did not add a fresh physical source that
   separates one first resident ask from 30 additional warm asks for
   `regression_tiny`, `ask_short`, and `ask_normal`. If that evidence is still
   needed, open or use a new narrow physical measurement issue instead of
   treating #1069 as open.
2. [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071)
   is closed by [#1208](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1208).
   The dense Rust GGUF thread/core matrix now covers current default,
   1-thread, 4-thread, and 8-thread variants with Windows power, AC/battery,
   thermal availability, utilization/frequency unavailability, fallback, and
   claim-boundary context. It does not justify thread tuning: default and
   `threads_1` both measured one effective thread, and `threads_4` /
   `threads_8` were slower across the measured profiles.
   [#1186](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186)
   is closed by [#1194](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1194),
   which added the smaller runner and receipt-builder contract needed to emit
   that matrix without adding CPU tuning, route policy, or speedup claims.
3. [#1156](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1156)
   landed the current comparison-qualification guard. Treat it as completed
   validation hardening, not as measurement evidence.
4. [#1201](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1201)
   is closed by [#1207](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1207).
   Treat it as source-receipt enrichment support for #1208, not as an open
   blocker.
5. [#1209](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1209)
   is closed as the post-matrix CPU slow-path review. Treat it as the decision
   that #1208 does not justify CPU optimization, default thread tuning,
   OpenVINO CPU promotion, or route-policy changes. The next CPU work should
   use [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232)
   for resident Rust GGUF phase attribution and no-reload evidence. Matched
   Rust GGUF CPU versus OpenVINO CPU comparison now uses
   [#1365](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1365);
   later affinity/topology work still needs a separate narrow issue once the
   evidence target is concrete.
6. [#1280](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1280)
   is closed by [#1334](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1334)
   as the physical resident run issue. The committed package records 33
   prompts, 32 warm asks after first, fallback false, model/tokenizer loaded
   once, quality passing, determinism passing, and
   `diagnostic_package_reviewable=true` while preserving
   `resident_phase_qualified=false` and `benchmark_qualified=false`.
7. [#1281](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1281)
   is closed by [#1290](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1290).
   It added prompt-render timing, quality-gate timing, detokenization summary
   exposure, and resident memory lifecycle support without route-policy,
   optimization, speedup, power, accelerator, or BitNet claims.
8. [#1291](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1291)
   is closed by [#1292](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1292)
   as the `receipt_write_ms` and `telemetry_ms` scope decision. Do not
   implement a two-pass receipt write, broad telemetry layer, or qualification
   relaxation until a later issue or spec defines different accepted receipt
   fields and summarizer rules.
9. [#1311](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1311)
   is closed by [#1319](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1319)
   as the resident status contract issue. Current receipts may distinguish
   diagnostic reviewability from strict `resident_phase_qualified` evidence
   when the only blockers are the #1291/#1292 `not_exposed` fields. #1334 uses
   that status split for the committed physical package. Future #1232 follow-up
   packages must preserve strict false qualification, benchmark false
   qualification, explicit contract-not-exposed blockers, and no route-policy
   or optimization claim unless a later contract revises the rule.

Do not start CPU optimization, default thread tuning, OpenVINO CPU promotion, or
route-policy changes from #1208. The matrix answers one platform question by
showing no thread-count win in the current host/corpus context; it does not
identify a safe runtime optimization target.

## Claim Boundary

This research does not add:

- new Lunar Lake inference;
- generated dashboards;
- route promotion;
- OpenVINO CPU promotion;
- CPU speedup claims;
- BitNet QK256/I2_S performance claims;
- low-power or battery evidence;
- kernel or tokenizer behavior changes.

It only reviews current CPU slow-path evidence and recommends small
measurement-first follow-up work.
