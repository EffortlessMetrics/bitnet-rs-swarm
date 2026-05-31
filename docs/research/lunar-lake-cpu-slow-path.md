# Lunar Lake CPU Slow-Path Research

Research issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1035

Research date: 2026-05-30

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
- existing receipts do not explain thread/core behavior on the 4 P-core plus
  4 low-power E-core 258V topology.

Do not start a CPU runtime optimization PR from this research alone. The next
implementation should be a small receipt or instrumentation PR that makes the
phase attribution harder to dispute.

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
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-resident-session.json` | Resident Rust GGUF CPU prompt loop, no model/tokenizer reload per prompt | Model loaded once; tokenizer loaded once; ask_short mean total 11158.750 ms; ask_normal mean total 16407.372 ms; no model or tokenizer reload observed |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-profile-run.json` | Explicit Rust GGUF CPU heavy-profile timing | prefill_heavy total 1373681.117 ms for 2734 prompt tokens and 16 generated tokens; decode_heavy total 123115.592 ms for 67 prompt tokens and 512 generated tokens |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-runtime-comparison.json` | Refreshed Rust GGUF CPU versus OpenVINO CPU diagnostic comparison | Rust resident ask_short mean 11158.750 ms and ask_normal mean 16407.372 ms; OpenVINO CPU corpus-v2 now passes 14/14 with fallback false, but the receipt remains context-only because model format, timing scope, prompt-render, tokenization, and matched-profile gaps block benchmark qualification |
| `ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json` | Newer OpenVINO CPU/GPU/NPU corpus-v2 receipt | OpenVINO CPU resolved to `Intel(R) Core(TM) Ultra 7 258V`, constructed in 981.455 ms, ran 14/14 corpus-v2 cases with fallback false and direct generated token IDs |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-cpu-corpus-v2-diagnosis.json` | OpenVINO CPU diagnosis | OpenVINO CPU corpus-v2 diagnosis says 14 total, 14 passed, 0 failed, no fallback, direct generated token IDs available |

The OpenVINO CPU comparison evidence was refreshed against the newer
corpus-v2 run. Do not use the runtime-comparison receipt as a benchmark
speedup claim: Rust GGUF CPU uses Q8_0 GGUF and OpenVINO CPU uses INT4_SYM
OpenVINO IR, timing scopes still differ, OpenVINO tokenization/detokenization
metrics are not fully exposed, and several corpus-v2 profiles lack matched
Rust resident evidence.

## Top Likely Causes

### 1. Rust GGUF CPU Prefill And First-Token Cost

This is the strongest current root-cause candidate for resident asks.

Resident CPU receipts remove per-prompt model and tokenizer reload, but the
route is still slow:

| Profile | Mean total | Mean time to first token | Mean prefill | Mean decode | Mean tokenize |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ask_short` | 11158.750 ms | 8531.800 ms | 7656.633 ms | 2950.161 ms | 466.206 ms |
| `ask_normal` | 16407.372 ms | 8834.200 ms | 7961.152 ms | 7899.989 ms | 465.554 ms |
| `regression_tiny` | 12314.790 ms | 9660.900 ms | 8737.639 ms | 3018.640 ms | 476.496 ms |

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

- resident `ask_short` generated 9 tokens at about 3.051 tokens/s;
- resident `ask_normal` generated 24 tokens at about 3.038 tokens/s;
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
- resident model load: 13592.773 ms;
- model SHA-256: 579.851 ms;
- tokenizer load: 568.949 ms;
- prompt count: 30;
- no per-prompt model or tokenizer reload observed.

Removing reload improved total latency. The cold-to-resident total ratio was
2.508 for `ask_short`, 1.706 for `ask_normal`, and 2.273 for
`regression_tiny`.

That is enough to justify resident-session measurement, but not enough to say
reload is the full root cause.

## Less Likely Or Unproven Causes

### Tokenizer And Template Setup

Tokenizer work is measurable but not dominant in current receipts:

- cold one-off tokenize: 482.325 ms;
- resident ask_short tokenize mean: 466.206 ms;
- resident ask_normal tokenize mean: 465.554 ms;
- resident tokenizer load: 568.949 ms.

This is worth cleaning up after the larger prefill/decode and cold-load
questions, but it is not the top slow-path cause.

### Receipt Overhead

The resident receipt enabled per-prompt receipts and recorded reusable buffers,
but no receipt currently isolates:

- JSON serialization time;
- prompt sidecar write time;
- stdout or logging time;
- quality-gate evaluation time;
- telemetry collection time.

Receipt overhead may matter for tight benchmark loops, but the existing
latencies are too large for receipt overhead to be the main explanation.

### Thread And Core Behavior

Thread/core behavior is a credible missing variable, not a proven root cause.

The 258V platform has 4 P-cores plus 4 low-power E-cores, and the roadmap says
power mode and thermal profile matter. The receipts do not yet record:

- P-core versus E-core affinity;
- effective thread count at runtime;
- CPU frequency or throttling;
- Windows power mode during each phase;
- per-thread CPU utilization;
- whether Rayon, OMP, MKL, or BLAS settings affect this path.

Do not tune thread counts blindly. First add a small matrix receipt that
records the above fields.

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

OpenVINO CPU profile timing in that receipt is far lower than Rust GGUF CPU
resident prompt-loop timing, for example:

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

### Thread/Core Matrix Receipt

Detailed plan: `docs/research/lunar-lake-cpu-thread-core-matrix.md`.

Run the same resident ask set across a small matrix:

| Variant | Purpose |
| --- | --- |
| current default | preserve baseline |
| `--threads 1` | isolate single-thread behavior |
| `--threads 4` | approximate P-core-only capacity without affinity proof |
| `--threads 8` | full logical 258V thread count |
| optional P-core affinity | test performance-core placement |
| optional E-core affinity | test low-power-core placement |

Required context:

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
| 1 | Add CPU phase attribution receipt fields and a fixture/test for schema validation | High: makes reload, prefill, decode, and receipt overhead separable | Low: docs/schema/unit-test surface |
| 2 | Add resident CPU session refresh receipt with per-prompt overhead accounting | High: confirms whether no-reload path is still prefill/decode bound | Medium: hardware run needed, but no route-policy change |
| 3 | Add thread/core matrix receipt generator | Medium-high: tests the most plausible platform-specific missing variable | Medium: may require Windows affinity and hardware scheduling care |
| 4 | Refresh Rust GGUF CPU versus OpenVINO CPU comparison | Medium: clarifies whether OpenVINO CPU is a route candidate or only diagnostic context | Medium: OpenVINO hardware/software run, but docs/receipt only |
| 5 | Optimize tokenizer/template setup | Low-medium: visible hundreds of milliseconds, but not dominant | Low-medium: local code change risk depends on tokenizer ownership |
| 6 | Change prefill/decode kernels or route policy | Potentially high, but evidence not yet precise enough | High: broad runtime and CI churn risk |

## Recommended Next Step

Open a small follow-up item, `CPU-SLM-PERF-001`, for a receipt/schema hardening
PR only:

```text
Add Lunar Lake Rust GGUF CPU slow-path phase attribution fields and tests.
No route-policy change. No kernel optimization. No generated dashboard refresh.
```

Acceptance:

- the receipt distinguishes cold model load, tokenizer load, prompt rendering,
  tokenize, prefill, first token, decode, detokenize, quality gate, receipt
  write, telemetry, and total response;
- unavailable sub-phases are marked `not_exposed` instead of `0`;
- profile, prompt tokens, generated tokens, thread count, power scheme, and
  fallback status are present;
- the committed docs say the new receipt is measurement infrastructure, not a
  CPU performance fix.

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
