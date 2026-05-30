# Lunar Lake NPU Cold-Start Research

Research issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1032

Research date: 2026-05-30

Repository: `EffortlessMetrics/bitnet-rs-swarm`

## Executive Summary

The current Lunar Lake NPU evidence supports one narrow conclusion:
OpenVINO NPU is useful only when the pipeline is already resident or the model
cache is already warm. It does not support promoting NPU for cold one-off asks,
`ask_short`, `ask_normal`, or `low_power`.

Committed receipts show:

- cold NPU startup is dominated by OpenVINO `LLMPipeline` construction,
  compile/load, device transfer, or cache-miss work;
- cold pipeline/load samples are 29.373s to 35.470s, with a 32.513s mean;
- hot generation after load is much smaller, generally about 0.49s to 0.52s
  for the bounded math/operator receipts;
- a second OpenVINO GenAI process using the same cache directory reduced
  pipeline construction from 28.104s to 0.873s;
- a same-process resident session completed 30/30 warm asks with
  `fallback_used=false`, no answer/token/route drift, 764ms mean generation
  wall time, and 221ms mean OpenVINO time to first token;
- the resident receipt also recorded about 692 MB of resident memory growth
  after pipeline construction and warm-loop execution.

That evidence is promising for a resident NPU profile, but it is not a broad
NPU route promotion, power-advantage, native BitNet, or low-power claim.

## Current Route Status

`lunar-lake-route-profile-comparison.json` already applies the right boundary:

- `dense_slm_openvino_npu_candidate` is promoted only for `warm_resident`;
- NPU remains candidate-only for `regression_tiny`, `ask_short`,
  `ask_normal`, `prefill_heavy`, `decode_heavy`, and `low_power`;
- `low_power` additionally remains blocked by missing power-advantage evidence;
- profile promotion is explicitly not an all-profile acceleration or power
  claim.

Keep that boundary until the measurements below are collected and reviewed.

## Evidence Map

| Receipt | Scope | Key NPU findings |
| --- | --- | --- |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cold-start-diagnosis.json` | Derived diagnosis, no new inference | `openvino_pipeline_load_or_device_compile_dominated`; 4 pipeline/load samples, 29.373s min, 32.513s mean, 35.470s max; operator load-to-generation ratio 57.59; phase-runner load-to-generation ratio 68.13 |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cache-experiment.json` | Two separate OpenVINO GenAI NPU processes, one cache dir | First pipeline construct 28.104s; second construct 0.873s; 0.031 second/first ratio; one 154,693,720-byte cache blob; answer gates passed both runs; no runtime cache-hit metric exposed |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-resident-session.json` | Same-process NPU pipeline plus repeated warm asks | Pipeline construct 1.470s with cache requested; 30/30 warm asks passed; warm generation mean 764ms, p95 1171ms; OpenVINO TTFT mean 221ms, p95 257ms; no answer/token/fallback/route drift |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-operator-ask-auto-npu-warm-resident-math-brief.json` | Operator ask for `warm_resident` route evidence | `profile_id=warm_resident`; selected backend `openvino-npu`; fallback false; answer gate passed; pipeline construct 32.032s and generation wall 862ms, so the receipt is not a cold one-off promotion |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cold-warm-profile-benchmark.json` | Profile benchmark index | NPU remains blocked outside `warm_resident`; `low_power` has no benchmark-qualified promoted route; warm resident timing excludes one-off NPU pipeline construction |

One anomaly should stay visible: the cold-start diagnosis copied a negative
`throughput_tokens_per_s` value from the corpus-v2 context. Do not use that
derived aggregate as NPU throughput evidence. Use direct generation wall,
OpenVINO TTFT, answer-gate, fallback, and generated-token fields instead.

## Cold-Start Phase Model

The existing receipts name the problem, but they do not fully split it. Treat
the NPU startup path as these phases:

1. Asset resolution and identity checks.
   Existing receipts record model, OpenVINO IR, tokenizer, and prompt-template
   identity, but not separate wall time for each lookup.
2. Tokenizer/config load and prompt-template setup.
   Current OpenVINO perf metrics record `tokenization.mean_ms=-1.0`, so they
   do not prove tokenizer setup cost. Add explicit host-side timers.
3. OpenVINO GenAI `LLMPipeline` construction.
   Current `pipeline_construct_wall_ms` covers this host-side block.
4. OpenVINO model compile/load/device transfer/cache-miss work.
   Current `load_time_ms` is close to `pipeline_construct_wall_ms`, but it does
   not split compile from device transfer or cache behavior.
5. First ask after construction.
   Existing receipts capture generation wall, streamed text chunk timing,
   OpenVINO time to first token, throughput, answer gates, and token IDs.
6. Same-process warm resident asks.
   Existing resident evidence proves stable repeated asks after one pipeline is
   held alive, not cold interactive startup.
7. Second-process cache reuse.
   Existing cache evidence shows the same cache directory can materially reduce
   later pipeline construction, but it does not expose a direct runtime cache
   hit metric.
8. `AUTO` selection and device routing.
   Existing NPU promotion should depend on explicit selected backend/device
   receipts. `AUTO` needs separate evidence for selected device and fallback.

## Cache Experiment Plan

Purpose: separate cold cache-miss construction from second-process cache reuse.

Run sequence:

1. Create a fresh, named cache directory under `target/openvino-cache`.
2. Record a pre-run cache snapshot: directory exists, file count, total bytes,
   file names, and optional hashes for small metadata files.
3. Start process A with explicit `device=NPU`, `CACHE_DIR`, fixed model,
   fixed prompt, fixed generation config, and answer gate.
4. Record process A pipeline construction, OpenVINO `load_time_ms`,
   generation wall, OpenVINO TTFT, throughput, generated token IDs,
   selected runtime/device, fallback status, and answer gate.
5. Record the after-A cache snapshot.
6. Start process B with the same cache directory, model, prompt, generation
   config, device, and answer gate.
7. Record the same process B fields and the after-B cache snapshot.
8. Classify cache effect by timing and stable cache files. Do not claim a
   direct OpenVINO cache-hit metric unless the runtime exposes one.

Required receipt fields:

- `comparison_scope`
- `cache_dir`
- `cache_enabled`
- `cache_writable`
- `cache_config_status`
- `cache_hit_runtime_metric_available`
- `initial_snapshot`
- `after_first_process_snapshot`
- `after_second_process_snapshot`
- `first_pipeline_construct_wall_ms`
- `second_pipeline_construct_wall_ms`
- `second_to_first_construct_ratio`
- `construct_improvement_ms`
- `openvino_load_time_ms` per process
- `runtime_api`
- `runtime_device`
- `resolved_device`
- `selected_backend`
- `selected_kernel_or_runtime`
- `fallback_used=false`
- `answer_gate_passed`
- `generated_token_ids_available_from_pipeline`
- OpenVINO version and relevant device properties

Acceptance for cache evidence:

- both process runs use explicit NPU and report `fallback_used=false`;
- both answer gates pass;
- generated token IDs are available from OpenVINO GenAI output;
- cache files are created by the first run and stable or reused by the second;
- second-process pipeline construction is materially lower than first-process
  construction;
- the receipt states that the cache classification is timing-derived unless a
  direct runtime cache-hit metric is present.

## Resident-Session Experiment Plan

Purpose: prove whether a long-lived NPU session can be a useful route target
without hiding the cold-start cost.

Run sequence:

1. Construct one explicit NPU `LLMPipeline`.
2. Capture memory before pipeline construction, after construction, after the
   first ask, and after the warm loop.
3. Run one cold first ask through the constructed pipeline.
4. Run at least 30 warm asks through the same pipeline.
5. Use a small fixed corpus with answer gates and stable prompt templates.
6. Record direct generated token IDs when OpenVINO GenAI exposes them.
7. Record answer drift, generated-token drift, fallback drift, and route drift.

Required receipt fields:

- `resident_session_ready`
- `same_process_pipeline_reused`
- `pipeline_construct_wall_ms`
- `cache_config_status`
- `cold_first_ask.ask_count`
- `cold_first_ask.generation_wall_ms`
- `cold_first_ask.openvino_time_to_first_token_ms`
- `warm_repeats_requested`
- `warm_resident_asks.ask_count`
- `warm_resident_asks.passed`
- `warm_resident_asks.failed`
- `warm_resident_asks.fallback_used=false`
- `warm_resident_asks.generation_wall_ms.{min,mean,p95,max}`
- `warm_resident_asks.openvino_time_to_first_token_ms.{min,mean,p95,max}`
- `warm_resident_asks.throughput_tokens_per_s.{min,mean,p95,max}`
- `answer_drift_detected=false`
- `generated_token_drift_detected=false`
- `fallback_drift_detected=false`
- `route_drift_detected=false`
- `memory_samples`
- `resident_memory_growth_bytes`

Acceptance for resident evidence:

- the same pipeline is reused across the warm loop;
- no fallback, answer drift, token drift, or route drift appears;
- warm timing is summarized separately from the cold first ask and pipeline
  construct;
- resident memory growth is recorded and bounded for the expected workflow;
- the receipt states that resident proof does not remove the one-off cold-start
  blocker.

## `AUTO` Device Plan

`AUTO` should not be used as a route-promotion shortcut.

Before any `AUTO`-based claim, collect a receipt that compares:

- explicit `openvino-npu`;
- explicit `openvino-gpu`;
- `AUTO`;
- the same model, prompt, generation config, answer gate, and cache settings.

Required fields:

- requested device;
- selected backend;
- OpenVINO runtime device;
- resolved device name;
- `EXECUTION_DEVICES` or equivalent selected-device property, if exposed;
- fallback status;
- answer gate;
- generated token IDs;
- pipeline construct and generation timings;
- cache directory and cache status.

Promotion can only use `AUTO` evidence if it proves the selected device and
fallback behavior for the target profile. If selected-device visibility is
missing, keep `AUTO` as diagnostic evidence only.

## Tokenizer And Asset Reload Gap

The current receipts identify the model and tokenizer artifacts, but they do
not prove whether tokenizer files, prompt templates, model metadata, or OpenVINO
IR assets are reloaded per ask.

Add host-side phase timers around:

- model path resolution;
- OpenVINO IR existence/hash lookup;
- tokenizer JSON/config lookup;
- tokenizer construction;
- prompt rendering;
- prompt tokenization;
- `LLMPipeline` construction;
- first `generate` call;
- repeated warm `generate` calls.

The goal is to separate Rust/host setup from OpenVINO compile/load work. Do not
rewrite route policy until that split exists.

## Promotion Rules

### `warm_resident`

Keep NPU promoted only for `warm_resident` when all of these remain true:

- profile selection is explicitly `warm_resident`;
- selected backend is `openvino-npu`;
- runtime device is `NPU`;
- fallback is false;
- resident session is ready;
- same-process pipeline reuse is true;
- warm loop answer gates pass;
- no answer, token, fallback, or route drift is detected;
- generated token IDs are available;
- warm timing excludes one-off pipeline construction;
- the claim boundary excludes broad speedup, power advantage, native BitNet,
  and all-profile acceleration claims.

### `ask_short` And `ask_normal`

Keep NPU blocked for `ask_short` and `ask_normal` while cold one-off pipeline
construction is about 29s to 35s and dominates total response time.

Promotion would need profile-matched evidence that either:

- the user workflow is explicitly resident or cache-warm before the ask starts;
  or
- total response time including construct/cache behavior beats the current
  promoted route with passing answer gates and no fallback.

Current receipts do not prove that.

### Cold One-Off Asks

Keep NPU blocked for cold one-off asks. The cache and resident receipts show
ways around startup cost, but they do not erase the startup cost for a fresh
uncached interactive process.

### `low_power`

Keep NPU blocked for `low_power` until `LNL258V-POWER-006` provides real
battery-mode route samples and energy-proxy evidence. Resident timing is not a
power-advantage claim.

### BitNet Claims

Do not treat dense Qwen OpenVINO NPU evidence as BitNet QK256/I2_S behavior
proof. It can guide Lunar Lake route ergonomics, but it does not prove native
BitNet NPU execution.

## Recommended Next Issues

1. `LNL258V-NPU-PHASE-001`: add phase timer schema for tokenizer, asset,
   `LLMPipeline`, compile/load, first ask, and warm ask timing.
2. `LNL258V-NPU-CACHE-002`: rerun the cache experiment with explicit OpenVINO
   version, NPU device properties, and cache snapshots that include stable file
   hashes where practical.
3. `LNL258V-NPU-RESIDENT-003`: expose an operator-facing resident-session
   measurement command and receipt schema.
4. `LNL258V-NPU-AUTO-001`: collect `AUTO` selected-device evidence before any
   route-policy use of `AUTO`.
5. `LNL258V-ROUTE-REVIEW-001`: review route policy only after the phase, cache,
   resident, and power evidence is current.

## Claim Boundary

This document is research and issue shaping only.

It does not add:

- new Lunar Lake inference;
- new generated dashboards;
- new route policy;
- route promotion;
- speedup evidence;
- power-advantage evidence;
- native OpenCL or native NPU proof;
- BitNet QK256/I2_S behavior proof.

It documents the current NPU cold-start evidence, the remaining phase gaps, and
the acceptance criteria for future small PRs.

## References

- OpenVINO documentation: model caching overview,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html
- OpenVINO documentation: GenAI inference on NPU,
  https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- OpenVINO documentation: `LLMPipeline` API,
  https://docs.openvino.ai/2024/api/genai_api/_autosummary/openvino_genai.LLMPipeline.html
- OpenVINO documentation: NPU device,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- OpenVINO documentation: query device properties,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html
- OpenVINO documentation: automatic device selection,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html
