# BITNET-SPEC-OPENVINO-PHASE-TIMING: OpenVINO Route Phase Timing Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1139](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1139), [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1189](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1189)
Linked PRs: [#1141](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1141), [#1191](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1191)
Support-tier impact: no promotion; defines phase timing receipt gates
Policy impact: no policy exception

## Purpose

Define the phase timing evidence required before OpenVINO CPU, GPU, or NPU
dense SLM routes can be compared or promoted for a named Lunar Lake workload
profile.

This spec does not run benchmarks, promote routes, claim speedup, claim power
advantage, prove broad dense SLM quality, or prove BitNet QK256/I2_S behavior.

## Timing Profiles

Timing receipts must identify the workload profile they cover:

| Profile | Prompt bound | Output bound | Primary question |
| --- | ---: | ---: | --- |
| `regression_tiny` | <= 64 tokens | <= 16 tokens | cheap sanity route |
| `ask_short` | <= 64 tokens | <= 32 tokens | one-off short ask |
| `ask_normal` | <= 512 tokens | <= 128 tokens | default local ask |
| `prefill_heavy` | >= 2048 tokens | <= 64 tokens | long prompt cost |
| `decode_heavy` | <= 256 tokens | >= 512 tokens | long answer cost |
| `structured` | profile-specific | profile-specific | constrained output cost |
| `low_power` | profile-specific | profile-specific | energy or power tradeoff |
| `warm_resident` | profile-specific | profile-specific | resident route latency |

Timing evidence applies only to the profile whose prompt and output token bounds
it satisfies. A 9-token smoke ask cannot qualify `ask_normal` unless the receipt
records an approved proxy policy; proxy evidence cannot promote a route.

## Required Receipt Fields

Minimum timing receipt shape:

```json
{
  "artifact_kind": "openvino_phase_timing",
  "route_id": "openvino_dense_slm_gpu_arc140v",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "fallback_used": false,
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "profile": "ask_normal",
  "profile_applicability": {
    "prompt_token_count": 384,
    "generated_token_count": 96,
    "fits_prompt_bound": true,
    "fits_output_bound": true,
    "proxy_only": false
  },
  "cold_warm_mode": "first_ever_cold|cached_cold_process|warm_same_process|resident_session",
  "phase_ms": {
    "process_start": null,
    "pipeline_construct": null,
    "model_read": null,
    "compile_model": null,
    "cache_lookup": null,
    "tokenizer_load": null,
    "prompt_render": null,
    "tokenize": null,
    "prefill": null,
    "first_token": null,
    "decode": null,
    "total_response": null
  },
  "throughput": {
    "decode_tokens_per_second": null,
    "total_tokens_per_second": null
  },
  "telemetry": {
    "power_source": "AC|battery|unknown",
    "power_profile": "<profile-or-unknown>",
    "memory_available_mb": null,
    "thermal_context": "measured|unavailable"
  },
  "comparison": {
    "cpu_reference_artifact": "<path>",
    "beats_cpu_total_response": false,
    "beats_cpu_decode": false,
    "power_advantage_proven": false
  }
}
```

If a phase cannot be measured directly, the receipt must mark it as
`not_exposed`, `not_applicable`, or `derived` with the derivation source.
Missing phase timing must not be filled with a zero.

## Phase Definitions

- `process_start`: CLI or harness startup cost, when included in the measured
  route.
- `pipeline_construct`: OpenVINO GenAI `LLMPipeline` construction.
- `model_read`: exported model and tokenizer files read from disk.
- `compile_model`: OpenVINO compile or device compile.
- `cache_lookup`: cache lookup and cache-hit/miss overhead.
- `tokenizer_load`: tokenizer object load and template metadata setup.
- `prompt_render`: chat-template rendering.
- `tokenize`: prompt tokenization.
- `prefill`: prompt processing before the first generated token.
- `first_token`: latency from generate call or prefill end to first token or
  first decoded chunk, with source recorded.
- `decode`: generation after first token through stop condition.
- `total_response`: user-visible total for the declared timing scope.

The timing scope must say whether `total_response` includes process startup,
pipeline construction, model compile, cache lookup, and tokenizer load.

## NPU Phase Ownership

For `LNL258V-NPU-PHASE-001`, an NPU phase-timing receipt must assign each
timer to exactly one owner or mark it unavailable. The receipt must not use a
coarse timer as proof for a narrower internal OpenVINO phase unless the source
is explicit. #1191 implements the current host-owned receipt-builder and
validator guard for these value/status/source fields; that implementation is
schema support only and does not create new physical timing evidence or route
promotion.

| Receipt field | Owner and timing scope | Accepted source | If unavailable |
| --- | --- | --- | --- |
| `asset_resolution_wall_ms` | Host-side model, tokenizer, template, cache-dir, and artifact path resolution before pipeline construction | Harness wall-clock timer around path and metadata lookup | `not_exposed` |
| `model_metadata_or_hash_wall_ms` | Host-side model metadata, file stat, or hash work before the OpenVINO pipeline call | Harness wall-clock timer around the metadata or hash operation | `not_exposed` |
| `tokenizer_load_or_construct_wall_ms` | Host-side tokenizer object load, tokenizer metadata setup, or template asset load | Harness wall-clock timer around tokenizer construction | `not_exposed` |
| `prompt_render_wall_ms` | Host-side chat-template rendering before tokenization or generation | Harness wall-clock timer around prompt rendering | `not_exposed` |
| `prompt_tokenize_wall_ms` | Host-side prompt tokenization with the tokenizer used for the route sample | Harness wall-clock timer around encode/tokenize; OpenVINO perf metrics only when directly exposed and non-sentinel | `not_exposed` |
| `pipeline_construct_wall_ms` | Coarse `LLMPipeline` construction envelope, including unresolved model read, cache lookup, compile/load, model transfer, and runtime setup unless narrower fields are directly exposed | Harness wall-clock timer around pipeline construction | `not_exposed` |
| `openvino_load_or_compile_wall_ms` | Direct OpenVINO model load, compile, or device-compile timing | Runtime metric or runtime log that names the load/compile phase | `not_exposed` |
| `cache_lookup_wall_ms` | Direct OpenVINO cache lookup or cache-hit/miss overhead | Runtime metric, runtime log, or stable cache-file evidence with source recorded | `not_exposed` |
| `cache_hit_status` | Cache hit, miss, reuse, or unknown classification for the exact model/export/runtime/device/config tuple | Runtime metric, runtime log, stable file reuse, or `timing_derived` diagnostic classification | `not_exposed` |
| `first_generate_wall_ms` | First `generate` call after pipeline construction, including prefill, first token, and decode if narrower splits are unavailable | Harness wall-clock timer around the first generate call | `not_exposed` |
| `ttft_ms` or `first_token_ms` | First visible token or first decoded chunk latency, with source and start point recorded | Runtime callback, stream chunk timing, or harness-visible first output timing | `not_exposed` |
| `decode_total_ms` | Generation after first token through stop condition | Runtime metric or harness timing that excludes first-token latency | `not_exposed` |
| `generation_wall_ms` | Total generation wall time from generate start to stop condition | Harness wall-clock timer around generate | `not_exposed` |
| `warm_repeat_summary` | Same-process or resident repeat timing after the setup ask | Per-request timers with request count, min, mean, p95, max, fallback, route, and device stability | `not_applicable` |
| `receipt_build_wall_ms` | Host-side receipt assembly after route timing is complete | Harness wall-clock timer around receipt construction | `not_exposed` |
| `receipt_write_wall_ms` | Host-side receipt serialization and file write | Harness wall-clock timer around write/flush | `not_exposed` |
| `telemetry_collect_wall_ms` | Host-side power, thermal, memory, and device telemetry collection | Harness wall-clock timer around telemetry probes | `not_exposed` |

OpenVINO GenAI timing values that use sentinel values such as `-1.0` must be
filtered out of measured summaries and recorded as unavailable. Sentinel values
must not be converted to zero, averaged into profile timing, or used to satisfy
profile applicability.

Timing-derived cache classification is diagnostic only. It may explain that a
second process was faster than a first process, but it is not direct runtime
cache-hit truth and cannot by itself promote NPU for cold/default asks,
`low_power`, or any power-advantage claim.

## Cold, Cache, Warm, and Resident Split

All NPU timing receipts must follow
`BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE`. GPU and CPU receipts should also
record cold and warm modes when available:

```text
first_ever_cold
cached_cold_process
warm_same_process
resident_session
```

A route can be promoted for a warm/resident profile without being promoted for
cold one-off asks, but the route reason must state the cold-start caveat.

## Profile Applicability

Each timing receipt must record:

- prompt token count;
- generated token count;
- requested `max_new_tokens`;
- actual stop reason;
- whether prompt/output counts satisfy the named profile;
- whether the evidence is direct or borrowed from another profile.

Promotion review must block timing evidence when:

- token counts are missing;
- evidence is proxy-only;
- timing was gathered under a different generation config;
- the route falls back;
- quality for the same profile fails.

## OpenVINO Metrics

If OpenVINO exposes runtime performance metrics, receipts should include them in
an `openvino_perf` block:

```json
{
  "openvino_perf": {
    "available": true,
    "source": "compiled_model|pipeline|runtime_log",
    "metrics": {}
  }
}
```

If metrics are unavailable through OpenVINO GenAI, receipts must record
`available=false`. A route can still use external wall-clock timing, but the
gap remains visible.

## Telemetry and Power Context

Timing receipts must include the current telemetry context or link to a
telemetry artifact:

- AC or battery;
- Windows power profile;
- memory available;
- thermal context or explicit unavailable reason;
- power or energy proxy when used for a low-power claim.

Low-power promotion requires either measured power/energy evidence or an
accepted proxy policy. Latency alone is not a power advantage.

## Comparison Requirements

A timing comparison must use the same:

- source model and exported artifact class;
- prompt profile;
- answer corpus case or benchmark case;
- generation config;
- cold/warm mode;
- fallback policy.

If OpenVINO CPU, GPU, or NPU uses a different model format from GGUF CPU, the
receipt must record that difference and compare only at the route/profile level,
not claim token-level engine parity unless direct token evidence exists.

## Promotion Inputs

Timing evidence can support promotion review only when:

1. The route passes the quality corpus for the exact profile.
2. `fallback_used=false`.
3. Prompt/output token counts satisfy the profile.
4. Phase timing includes enough split to explain the advantage.
5. CPU reference timing for the same profile exists.
6. Any power or low-power claim has telemetry evidence.
7. Known gaps do not contradict the promotion.

Timing alone cannot promote a route.

## Rejection Examples

| Evidence | Required decision |
| --- | --- |
| GPU total response beats CPU on a 9-token smoke ask only | No `ask_normal` promotion |
| NPU hot decode is fast but cold compile is included in selected profile | Block cold one-off promotion |
| Timing lacks prompt token count | Block profile applicability |
| Timing route has `fallback_used=true` | Reject as accelerator timing |
| OpenVINO metric unavailable but wall-clock timing exists | Use wall-clock with explicit metric gap |
| Dense SLM timing improves | Do not claim BitNet QK256/I2_S speedup |

## Acceptance

This spec is complete when it defines:

1. Required timing profiles and token-bound applicability.
2. Phase timing fields and timing-scope definitions.
3. Cold, cached, warm, and resident timing separation.
4. OpenVINO metric, telemetry, and power-context requirements.
5. Comparison and promotion inputs that block proxy-only speedup claims.
6. NPU timer ownership for host setup, tokenizer/template work, pipeline
   construction, compile/load/cache, first generate, warm repeats, receipt
   overhead, and telemetry collection.
7. Fail-closed handling for sentinel OpenVINO metrics and timing-derived cache
   classification.
