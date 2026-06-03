# Lunar Lake NPU Cold-Start Research

Original research issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1032
Current cold/cache parent: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119
Direct cache-hit truth child: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1371
Closed cache rerun issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1160
Cache rerun closeout PR: https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1174
Phase-timing schema closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1139 / PR #1141
Receipt-validation alignment closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1143 / PR #1145
Cache-classification guard closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1154
AUTO selected-device issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149
AUTO selected-device review: [lunar-lake-openvino-auto-selected-device.md](../reviews/lunar-lake-openvino-auto-selected-device.md)
Host phase timing receipt guard closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1189 / PR #1191
AUTO debug-log evidence validator closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1216 / PR #1217
AUTO debug-log parser integration closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1242 / PR #1248
AUTO debug-log capture source integration closed by: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1251 / PR #1252
AUTO debug-log warning-boundary hardening closed by: PR #1254
OpenVINO generated-token visibility watch: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244 / PR #1350 / PR #1355
Original token visibility review issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123 / PR #1138
Route-policy watch issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1245 / PR #1358

Research date: 2026-05-30
Watch refresh: 2026-06-02
Cache-truth child refresh: 2026-06-03

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
- the historical cache experiment reduced a second OpenVINO GenAI process using
  the same cache directory from 28.104s to 0.873s, and the #1160 / #1174 rerun
  records a smaller but still material reduction from 11.164s to 0.945s;
- the 2026-06-01 current-main cache probe records a first explicit-NPU
  process at 10.839s pipeline construction and a second process using the same
  cache directory at 0.893s, with passing answer gates, `fallback_used=false`,
  direct generated-token IDs, timing-derived cache evidence, and no direct
  runtime cache-hit metric exposed;
- a same-process resident session completed 30/30 warm asks with
  `fallback_used=false`, no answer/token/route drift, 764ms mean generation
  wall time, and 221ms mean OpenVINO time to first token;
- a paired current-main resident diagnostic using the same cache directory
  completed 10/10 warm asks with `fallback_used=false`, no answer/token/route
  drift, 315ms mean generation wall time, and 161ms mean OpenVINO time to first
  token. Treat it as diagnostic cache/resident context, not as a replacement
  for the 30/30 warm-resident acceptance receipt;
- the resident receipt also recorded about 692 MB of resident memory growth
  after pipeline construction and warm-loop execution.
- #1189 / #1191 added the host-side `host_phase_timing` receipt surface and
  validator guard for future phase receipts. That is schema and guard support;
  it is not new physical NPU evidence and does not by itself promote a route.
- #1216 / #1217 added machine-readable validator support for the block-scoped
  OpenVINO GenAI `AUTO` debug-log evidence artifact. That is selected-device
  review support for one stateful LLM model block; it is not generated
  phase-receipt selected-device proof, route-policy evidence, or NPU promotion.
- #1242 / #1248 and #1251 / #1252 added the parser helper and repeatable
  capture wrapper for that block-scoped OpenVINO GenAI `AUTO` debug-log
  evidence shape. This is receipt-source support only; it is not a route-policy
  change, NPU promotion, `low_power` evidence, or a power/speedup claim.
- #1254 preserved SDPA warning and AUTO startup/running fallback-disabled line
  references when those lines appear in the raw debug log. That keeps the
  diagnostic context reviewable, but it does not change the paired phase
  receipt's application `fallback_used=false` decision or create selected-device
  proof.
- #1244 is the live watch issue for future OpenVINO generated-token visibility
  schema or checker gaps. PR #1350 and PR #1355 anchor the current shared
  direct-token helper and fail-closed validator path, so #1244 remains
  watch-only unless a future receipt bypasses that helper or exposes ambiguous
  direct/proxy/unavailable token status. The original token visibility review
  issue #1123 is closed; NPU cache and resident evidence still need direct
  generated-token IDs for promotion-grade token-drift claims, but token
  visibility alone does not change cold/cache, `low_power`, or route-policy
  blockers.

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
The #1245 route-policy watch was refreshed by #1358 with the same boundary:
future NPU route-policy work still needs a linked evidence finding before any
keep, conditional, narrow, revoke, or blocked decision changes the ledger.

## Evidence Map

| Receipt | Scope | Key NPU findings |
| --- | --- | --- |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cold-start-diagnosis.json` | Derived diagnosis, no new inference | `openvino_pipeline_load_or_device_compile_dominated`; 4 pipeline/load samples, 29.373s min, 32.513s mean, 35.470s max; operator load-to-generation ratio 57.59; phase-runner load-to-generation ratio 68.13 |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cache-experiment.json` | Two separate OpenVINO GenAI NPU processes, one cache dir | First pipeline construct 28.104s; second construct 0.873s; 0.031 second/first ratio; one 154,693,720-byte cache blob; answer gates passed both runs; no runtime cache-hit metric exposed |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cache-rerun-20260601.json` | Closed #1160 / #1174 two-process NPU cache rerun, one cache dir | First pipeline construct 11.164s; second construct 0.945s; 0.085 second/first ratio; 10.219s improvement; one 158,052,779-byte cache blob stayed stable; answer gates passed both runs; fallback false; direct generated token IDs available; direct runtime cache-hit metric not exposed |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cache-probe-20260601T1323Z.json` | Current-main two-process NPU cache diagnostic, one cache dir | First pipeline construct 10.839s; second construct 0.893s; 0.082 second/first ratio; 9.946s improvement; cache file stayed stable; answer gates passed both runs; fallback false; direct generated-token IDs available; direct runtime cache-hit metric not exposed |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-resident-session.json` | Same-process NPU pipeline plus repeated warm asks | Pipeline construct 1.470s with cache requested; 30/30 warm asks passed; warm generation mean 764ms, p95 1171ms; OpenVINO TTFT mean 221ms, p95 257ms; no answer/token/fallback/route drift |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-resident-session-20260601T1325Z.json` | Current-main same-process NPU resident diagnostic using the cache-probe directory | Pipeline construct 0.865s with cache requested; 10/10 warm asks passed; warm generation mean 315ms, p95 468ms; OpenVINO TTFT mean 161ms, p95 176ms; no answer/token/fallback/route drift; diagnostic only, not a new route-promotion receipt |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-operator-ask-auto-npu-warm-resident-math-brief.json` | Operator ask for `warm_resident` route evidence | `profile_id=warm_resident`; selected backend `openvino-npu`; fallback false; answer gate passed; pipeline construct 32.032s and generation wall 862ms, so the receipt is not a cold one-off promotion |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cold-warm-profile-benchmark.json` | Profile benchmark index | NPU remains blocked outside `warm_resident`; `low_power` has no benchmark-qualified promoted route; warm resident timing excludes one-off NPU pipeline construction |

One anomaly should stay visible: the cold-start diagnosis copied a negative
`throughput_tokens_per_s` value from the corpus-v2 context. Do not use that
derived aggregate as NPU throughput evidence. Use direct generation wall,
OpenVINO TTFT, answer-gate, fallback, and generated-token fields instead.

## Current Command Surface

The current command surface is enough to preserve the NPU cold/cache boundary,
but it does not yet expose every phase needed to close #1119.

- `scripts/openvino_genai_npu_cache_probe.py` measures two explicit NPU child
  processes sharing one cache directory. It records coarse `LLMPipeline`
  construction and generation timing, cache snapshots, answer gates, fallback
  status, and generated token IDs. It does not expose a direct runtime cache-hit
  metric.
- `scripts/openvino_genai_phase_receipt.py` records bounded CPU/GPU/NPU phase
  evidence with direct generated token IDs, OpenVINO `PerfMetrics`, coarse
  `pipeline_construct_wall_ms`, streamer timing, generation timing, and the
  #1191 `host_phase_timing` block for host-owned setup, tokenizer/template,
  prompt render/tokenization, first-generate, receipt-overhead, telemetry, and
  explicit unavailable runtime phase statuses.
- `scripts/openvino_genai_token_utils.py` can now return host prompt render and
  prompt tokenization timing to the phase receipt path when requested. These
  host timers do not expose direct OpenVINO runtime cache-hit, compile, or load
  truth.
- `scripts/openvino_genai_npu_resident_session.py` records same-process
  warm-resident behavior. That evidence can support a `warm_resident` review,
  but it does not make cold one-off, default, or `low_power` NPU routing valid.
- #1149 owns the separate runtime-layer OpenVINO `AUTO` selected-device gap.
  Generic device visibility, CLI route selection, or explicit-NPU cache
  evidence should not be treated as `AUTO` execution proof.
- `scripts/openvino_genai_auto_debug_log_capture.py` is now the narrow source
  path for future #1149 debug-log captures. Use it only for concrete evidence
  packages or review decisions; do not use source availability as NPU cold/cache
  or route-policy proof.

The #1189 / #1191 host-side phase surface closes the immediate schema/guard
gap. The next #1119 implementation should only proceed if a physical evidence
run or runtime audit exposes a direct cache metric/log, an `AUTO`
selected-device field, a battery evidence field, or another specific missing
receipt boundary. It should not be another generic cache rerun, route-policy
mutation, benchmark matrix, or low-power promotion attempt.

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

The #1160 / #1174 rerun artifact
`ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cache-rerun-20260601.json`
satisfies this as a diagnostic cache snapshot: it records explicit NPU routing,
model/export/tokenizer identity, OpenVINO 2026.2.0 / GenAI 2026.2.0.0 context,
cache snapshots, first/second process phase summaries, passing answer gates,
direct generated token IDs, and `fallback_used=false`. It still records
`cache_evidence_source=timing_derived` and
`direct_runtime_cache_hit_status.available=false`, so it remains diagnostic
cache evidence rather than direct runtime cache-hit truth or route-promotion
evidence.

Implementation note: `bitnet lunar-lake npu-cold-start-diagnosis` should ingest
the NPU cache experiment and expose a `cold_load_decomposition` block. That
block distinguishes the first-process cache miss from second-process cache
reuse, records required NPU timing and answer-gate fields for each run, and
marks missing direct cache-hit metrics as timing-derived rather than runtime
truth.

## OpenVINO Cache Source Boundary

OpenVINO's current NPU documentation supports cache configuration and cache
provenance as receipt fields, but it does not by itself expose a direct
promotion-grade cache-hit truth signal for the OpenVINO GenAI `LLMPipeline`
receipts used by this lane.

Relevant source boundary:

- GenAI NPU docs describe `CACHE_DIR` as the preferred device-neutral compiled
  model cache mechanism and keep `NPUW_CACHE_DIR` as the older NPU-specific
  option. They also describe `EXPORT_BLOB` / `BLOB_PATH` ahead-of-time import
  and export flows. These are configuration and provenance sources, not
  evidence that a specific receipt observed a runtime cache hit.
- NPU device docs describe two cache layers: UMD dynamic model caching and
  OpenVINO model caching via `ov::cache_dir`. They explain that a later cache
  hit imports a model instead of recompiling it, but the docs do not identify a
  read-only GenAI receipt field that reports "this run hit the cache".
- Query-device-property docs make `ov::supported_properties` the way to inspect
  available property names and mutability. Future receipt-source work should
  query and record the supported-property set before treating any cache or
  selected-device field as available.
- OpenVINO 2026 release notes add useful cache/import provenance and
  compatibility fields, including runtime requirements, compatibility checks,
  compiler-version traceability, and cache/export support details. Those fields
  can harden future receipts, but they still do not replace a direct cache-hit
  metric or accepted runtime log.

Current #1119 consequence:

- Keep committed cache receipts classified as `timing_derived` or
  file-reuse-derived when `direct_runtime_cache_hit_status.available=false`.
- #1371 is the current narrow child for direct OpenVINO GenAI NPU cache-hit
  truth. Use it only for one of these source shapes: a documented runtime
  cache-hit property, a parseable OpenVINO/NPU runtime log that distinguishes
  import-from-cache from compile, explicit `not_exposed` cache-hit truth, or
  provenance hardening for cache path/blob/compiler/compatibility metadata.
- Do not open another generic cache rerun, cold/default NPU route PR,
  `low_power` promotion, speedup/power claim, native NPU claim, or BitNet
  QK256/I2_S claim from cache configuration support alone.

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

The current fail-closed contract is recorded in
[lunar-lake-openvino-auto-selected-device.md](../reviews/lunar-lake-openvino-auto-selected-device.md).
That review distinguishes CLI `--device auto` route selection from OpenVINO
runtime-layer `AUTO` selected-device proof.

The 2026-06-01 runtime `AUTO` diagnostic package now compares:

- explicit `openvino-npu`;
- explicit `openvino-gpu`;
- `AUTO`;
- the same model, prompt, generation config, answer gate, and cache settings.

It records the required boundary fields:

- requested device;
- selected backend;
- OpenVINO runtime device;
- resolved device name;
- `EXECUTION_DEVICES` status as `not_exposed`;
- fallback status;
- answer gate;
- generated token IDs;
- pipeline construct and generation timings;
- cache directory and cache status.

The original diagnostic receipts validate and preserve the fail-closed result:
runtime `AUTO` can be requested and can pass the bounded answer gates, but the
public Python OpenVINO GenAI receipt source still cannot expose which execution
device the AUTO plugin selected internally through a normal property accessor.

The #1212 debug-log follow-up adds a narrower source for the same phase tuple:
`ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-auto-debug-log-evidence-20260601.json`
records `OPENVINO_LOG_LEVEL=2` stdout/stderr capture where the OpenVINO GenAI
stateful LLM model block prints `EXECUTION_DEVICES: GPU.0` and resolves it to
`Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)`. The same generated phase receipt
still records `selected_device_visibility_status=not_exposed` because the
script has no public GenAI property accessor.

Promotion can only use `AUTO` evidence if it proves the selected device and
fallback behavior for the target profile. The #1214 review update accepts the
debug-log source only as block-scoped selected-device visibility for the
observed stateful LLM model block. #1217 makes that evidence artifact
machine-readable and validator-admitted when it preserves the debug-log source,
log provenance, runtime `AUTO` scope, stateful LLM model block, parsed execution
devices, block applicability, paired answer/fallback status, and claim
boundary. It does not wire selected-device visibility into the generated phase
receipt, promote `AUTO`, promote NPU, change low-power routing, prove cold
one-off routing, claim power advantage, claim speedup, prove native accelerator
execution, or prove BitNet QK256/I2_S behavior.

PR #1254 adds warning-context hardening for the same debug-log source: SDPA
attention-backend warning lines and AUTO fallback-disabled startup/running
lines stay addressable when present in the raw log. Those warning references
help future reviews explain the diagnostic environment, but they are not an
application fallback override and they do not convert `AUTO` debug logs into
route-policy proof.

## Tokenizer And Asset Reload Gap

The committed historical receipts identify the model and tokenizer artifacts,
but they do not prove whether tokenizer files, prompt templates, model
metadata, or OpenVINO IR assets are reloaded per ask. The #1191
`host_phase_timing` receipt surface can now record the host-owned portions of
that split for future phase runs.

Future evidence runs should populate or explicitly mark unavailable:

- model path resolution;
- OpenVINO IR existence/hash lookup;
- tokenizer JSON/config lookup;
- tokenizer construction;
- prompt rendering;
- prompt tokenization;
- `LLMPipeline` construction;
- first `generate` call;
- repeated warm `generate` calls.

The goal remains to separate Rust/host setup from OpenVINO compile/load work.
Do not rewrite route policy from schema support alone; use the split only after
a fresh receipt records the relevant measured or unavailable fields.

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

## Current Issue Ownership

- #1119 remains the cold/cache research parent and keeps timing-derived cache
  classification diagnostic until a stricter policy is accepted.
- #1371 owns the direct cache-hit truth follow-up. It should distinguish
  `runtime_metric`, `runtime_log`, `file_reuse`, `timing_derived`, and
  `not_exposed` evidence, keep first-process and second-process phase timing
  separate, and fail closed if timing/file-derived evidence is labeled as a
  direct runtime cache hit.
- #1139 closed with the phase-timing schema contract for host setup,
  tokenizer/template setup, `LLMPipeline`, compile/load/cache behavior, first
  ask, warm asks, and receipt overhead.
- #1143 closed with #1145 after the committed OpenVINO NPU diagnosis and
  route-promotion receipts were aligned for direct validation without changing
  NPU promotion scope.
- #1154 closed because existing validator coverage already fails closed when
  timing-derived cache diagnostics are treated as direct runtime cache-hit truth
  without direct runtime evidence fields.
- #1149 owns `AUTO` selected-device evidence before any route-policy use of
  `AUTO`. The 2026-06-01 diagnostic package proves runtime `AUTO` can be
  requested and answer-gated, and the #1212 debug-log capture proves the
  stateful LLM model block executed on `GPU.0` in that one diagnostic run.
  #1217 validator-admits that debug-log evidence artifact under the same
  block-scoped boundary, but the generated phase receipt still records
  selected-device visibility as `not_exposed`; the current review contract is
  recorded in
  [lunar-lake-openvino-auto-selected-device.md](../reviews/lunar-lake-openvino-auto-selected-device.md);
  #1242 is closed by #1248 after landing the narrower parser helper for accepted
  `genai_debug_log` fields, and #1251 is closed by #1252 after landing the
  repeatable capture wrapper. PR #1254 then preserves SDPA warning and AUTO
  fallback-disabled line references without changing generated phase-receipt
  selected-device proof or application fallback status. Future #1149 work
  should use that wrapper only for materially useful evidence packages,
  API-bridge checks, or route review after the other gates exist. #1119 remains
  the broader cold/cache parent.
- #1120 closed with the warm-resident acceptance rule defined in
  [lunar-lake-npu-warm-resident-acceptance.md](../reviews/lunar-lake-npu-warm-resident-acceptance.md).
- #1162 closed with #1163 after the route diagnostics guard started failing
  closed when future `warm_resident` resident-session receipts omit the #1120
  pipeline, cold-first-ask, warm-loop, drift, token, memory, telemetry, or
  claim-boundary evidence.
- #1160 closed with #1174 after adding the current 2026-06-01 cache-rerun
  evidence package. The artifact is useful diagnostic evidence, but it does
  not close direct runtime cache-hit, cold/default NPU promotion, `AUTO`, or
  low-power evidence gaps.
- #1189 closed with #1191 after adding the `host_phase_timing` receipt-builder
  and validator guard. This closes the immediate host phase schema/guard gap,
  but it does not create new physical NPU evidence, direct runtime cache-hit
  truth, `AUTO` selected-device proof, or `low_power` evidence.
- #1064 remains the only path to `low_power` battery and energy evidence.

## Recommended Next Issues

1. `LNL258V-NPU-COLD-001` (#1119): keep broader cold/cache decomposition
   research open for fresh evidence that uses the #1191 host phase surface,
   direct runtime cache metrics/logs if exposed, or newly found phase/cache
   gaps. Do not reopen a generic schema or cache-rerun PR from #1119 alone.
2. `LNL258V-NPU-AUTO-001` (#1149): do not repeat a generic runtime `AUTO`
   rerun or generic schema/validator PR. #1217 already admits the current
   `genai_debug_log` evidence artifact, and #1248 added the parser helper for
   captured debug logs. #1252 added the repeatable capture wrapper for that
   accepted evidence shape, and #1254 preserved warning/internal-policy line
   references without changing the selected-device or fallback proof boundary.
   Open other #1149 work only to persist a materially useful wrapper-generated
   evidence package, test a public GenAI API or equivalent lower-level OpenVINO
   selected-device bridge for the same tuple, or run a route review after the
   exact selected-device, quality, timing, fallback, profile, and power gates
   exist. Use the contract in
   [lunar-lake-openvino-auto-selected-device.md](../reviews/lunar-lake-openvino-auto-selected-device.md)
   before any route-policy use of `AUTO`.
3. Direct cache-hit truth follow-up should use #1371. Open work there only for
   a newly exposed direct cache metric, parseable runtime log, stricter missing
   field, explicit unavailable truth-source handling, or precise
   cache/blob/compiler/compatibility provenance hardening. Do not reopen #1160
   as another generic rerun or repeat the #1189 host phase guard.
4. `LNL258V-ROUTE-REVIEW-001`: review route policy only after the phase, cache,
   resident, and power evidence is current.

## Claim Boundary

This document is research and issue shaping only.

It does not add:

- route-policy inference beyond the cited #1160 diagnostic cache rerun receipt;
- new generated dashboards;
- new route policy;
- route promotion;
- speedup evidence;
- power-advantage evidence;
- native OpenCL or native NPU proof;
- BitNet QK256/I2_S behavior proof.

It documents the current NPU cold-start evidence, the remaining evidence gaps,
and the acceptance criteria for future small PRs.

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
- OpenVINO documentation: release notes,
  https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html
