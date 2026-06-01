# Lunar Lake CPU Thread/Core Matrix Plan

Research issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071

Runner contract issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186

Source-receipt issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1201

Research date: 2026-05-31

Repository: `EffortlessMetrics/bitnet-rs-swarm`

## Executive Summary

The Lunar Lake dense Rust GGUF CPU route does not yet have thread/core matrix
evidence.

Existing artifacts prove useful surrounding facts, but they do not satisfy
`LNL258V-CPU-SLM-PERF-004`:

- `lunar-lake-cpu-slm-resident-session.json` records resident dense Qwen timing
  with `thread_count=1` and no sampled power scheme or AC/battery state.
- `lunar-lake-cpu-profile-run.json` records dense Rust GGUF heavy-profile
  timing, but not a thread-count matrix or scheduler context.
- `cpu-bitnet-perf-002-i2s-tiling-matrix.json` and
  `cpu-bitnet-perf-003-i2s-applied-thread-matrix.json` are BitNet
  QK256/I2_S microbench receipts. They are not dense Qwen GGUF resident-route
  evidence.

Do not tune CPU defaults, pin affinity by default, or claim a CPU speedup from
the current evidence. The next implementation should be a small measurement
receipt/harness that records thread count, Windows scheduling context, power
state, and resident dense Qwen phase timing across a fixed matrix.

[#1186](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186)
owns that narrow runner and receipt-builder contract; #1071 remains the
physical matrix evidence issue.
[#1201](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1201)
now owns the narrower source-receipt question: what measurement wrapper or
receipt-enrichment path should produce per-variant resident-session receipts
that the #1194 matrix builder can validate.

## Current Evidence Map

| Artifact | What It Proves | Why It Is Not Enough |
| --- | --- | --- |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-resident-session.json` | Dense Rust GGUF Qwen resident loop has no per-prompt model/tokenizer reload and records `thread_count=1`. | Only one thread context is represented; power scheme and AC/battery state are `not_sampled_in_slm_cpu_warm_session`; no affinity, utilization, or frequency proxy is present. |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-profile-run.json` | Dense Rust GGUF Qwen prefill-heavy and decode-heavy profile timings exist with fallback false. | It is not a resident multi-variant matrix and does not record requested/effective thread count, affinity mask, power mode, thermal, utilization, or frequency context. |
| `ci/hardware/intel-258v/2026-05-08/cpu-bitnet-perf-002-i2s-tiling-matrix.json` | QK256/I2_S tiling/thread candidates were recorded for BitNet microbench work. | Thread counts are recorded for a synthetic BitNet kernel search surface, not dense Qwen GGUF resident asks. |
| `ci/hardware/intel-258v/2026-05-08/cpu-bitnet-perf-003-i2s-applied-thread-matrix.json` | Scoped worker thread counts were applied inside sampled QK256/I2_S microbenches. | It explicitly does not claim full decode behavior and does not cover the dense SLM user-facing CPU path. |

## Current Command Surface

`lunar-lake cpu-slm-thread-core-matrix` is the narrow receipt-builder surface
for this lane. It does not run inference or collect hardware samples. It
aggregates explicit per-variant resident-session receipts with:

```text
--variant default=<resident-session-json>
--variant threads_1=<resident-session-json>
--variant threads_4=<resident-session-json>
--variant threads_8=<resident-session-json>
```

The builder emits `lunar-lake-cpu-slm-thread-core-matrix.json` and fails closed
when required variants, resident reuse, thread counts, power/AC-battery
context, telemetry statuses, fallback=false, generated-token ID source, or the
claim boundary are missing.

[#1186](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186)
owns that command and receipt-builder contract. #1071 remains the physical
matrix evidence issue.

Relevant existing surfaces do not close this gap:

- `lunar-lake cpu-slm-resident-session` summarizes an existing repeated
  warm-session receipt. It does not run a new thread/core matrix.
- `lunar-lake cpu-slm-thread-core-matrix` validates and aggregates one
  resident-session receipt per matrix variant. It does not create those source
  receipts.
- `lunar-lake ask --threads <N>` can produce one-off operator asks, but it is
  not enough for this issue if every variant reloads the model.
- `bitnet slm-warm-session --threads <N>` can produce dense GGUF warm-session
  work, but it is not sufficient for #1071 unless the source receipt also
  records the required thread identity, affinity, power, AC/battery,
  thermal/utilization/frequency, resident no-reload, token-visibility, and
  claim-boundary fields.
- BitNet QK256/I2_S thread matrix receipts are specialist kernel evidence, not
  dense Qwen GGUF resident-route evidence.

The next measurement surface must either run one resident session per variant
or explicitly mark `resident_session_reused=false` and fail the
resident-matrix acceptance. This is a measurement-command and receipt-source
gap, not a route-policy, CPU tuning, OpenVINO CPU promotion, or speedup issue.

## Source-Receipt Contract Gap

Issue #1194 closed the aggregate matrix-builder contract, but it did not create the
per-variant physical source receipts. #1201 is the current child issue for that
gap.

## Current Source-Surface Audit

Current-main code inspection on 2026-06-01 keeps #1201 as a source-receipt
contract issue, not a physical matrix issue.

`lunar-lake cpu-slm-thread-core-matrix` currently accepts only per-variant
source receipts with:

- `artifact_kind=lunar_lake_cpu_slm_resident_session`;
- required variants `default`, `threads_1`, `threads_4`, and `threads_8`;
- resident readiness, resident session reuse, `model_loaded_once=true`, and
  `tokenizer_loaded_once=true`;
- requested/effective thread count fields, thread environment capture, process
  affinity mask, honest affinity classification status, Windows power scheme,
  and AC/battery state;
- thermal availability, temperature status, CPU-utilization status, and
  frequency/throttle status as measured-or-unavailable fields;
- `regression_tiny`, `ask_short`, and `ask_normal` profiles with measured
  prompt-token counts, generated-token counts, direct generated-token source,
  passing answer gates, no fallback, and no model/tokenizer reload inside the
  resident loop;
- generated token IDs available from the source receipt, with the source marked
  as direct rather than retokenized decoded text or determinism-only stability;
- negative claim-boundary booleans for inference, route promotion, speedup,
  power advantage, acceleration, Arc/NPU execution, BitNet QK256/I2_S behavior,
  and hidden fallback.

Raw `bitnet slm-warm-session --threads <N>` receipts are therefore useful
measurement inputs, but they are not validator-ready matrix inputs by
themselves. The current aggregate receipt has `artifact_kind=slm_cpu_warm_session`,
records `cpu.threads`, and proves one model/tokenizer load for the warm-session
process, but it does not emit the `execution_context` block required by the
matrix builder, does not capture thread environment or process affinity, and
records power/thermal state as `not_sampled_in_slm_cpu_warm_session`.

The existing committed
`ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-resident-session.json`
has the accepted resident-session artifact kind and profile summaries, but it
is still only a single historical thread context. It does not satisfy #1071
because power scheme and AC/battery state are not measured, prompt-token counts
remain `not_exposed`, generated-token visibility is only determinism-group
stability unless a source is recorded explicitly, and default/1/4/8 variants do
not exist as separate source receipts.

The next implementation should therefore be a source wrapper or enrichment path
that emits validator-ready `lunar_lake_cpu_slm_resident_session` receipts per
variant. It can reuse `slm-warm-session` as the execution primitive, but it
must add the #1201 execution-context, telemetry, direct-token-source, and claim
boundary fields before the #1071 physical matrix run is attempted.

The next implementation should add or prove one narrow source surface before
the physical default / 1-thread / 4-thread / 8-thread matrix is attempted:

- a wrapper around `slm-warm-session` that records the required #1071 context
  per variant;
- an enrichment path for an existing resident-session receipt that adds the
  missing thread, power, telemetry, affinity, and unavailable-field statuses; or
- a fixture-only guard proving which source-receipt fields are mandatory before
  the aggregate builder accepts them.

Do not use #1201 to tune CPU defaults, promote OpenVINO CPU, compare speedups,
or run the physical matrix. It is the evidence-source contract needed before
issue #1071 can collect measurements that will pass the matrix validator.

## Required Receipt

Proposed path:

```text
ci/hardware/intel-258v/2026-05-08/lunar-lake-cpu-slm-thread-core-matrix.json
```

Proposed artifact kind:

```text
lunar_lake_cpu_slm_thread_core_matrix
```

The receipt should be generated only from physical Lunar Lake runs or from
per-variant source receipts that were produced by those runs. A plan-only
artifact must not be named as if it contains measurements.

## Matrix

Required variants:

| Variant ID | Requested Threads | Affinity |
| --- | ---: | --- |
| `default` | null | not set |
| `threads_1` | 1 | not set |
| `threads_4` | 4 | not set |
| `threads_8` | 8 | not set |

Optional variants:

| Variant ID | Requested Threads | Affinity |
| --- | ---: | --- |
| `p_core_affinity` | 4 | only if the mask and P-core classification are recorded accurately |
| `e_core_affinity` | 4 | only if the mask and E-core classification are recorded accurately |

Do not infer P-core or E-core placement from thread count alone. If Windows
does not expose a reliable core-class mapping during the run, record
`affinity_classification_status="not_exposed"` and omit the optional variants.

## Measurement Scope

Use the same dense Rust GGUF resident ask set across every variant:

- `regression_tiny`
- `ask_short`
- `ask_normal`

The existing `lunar-lake ask --threads <N>` surface is useful for one-off
operator asks, but it is not sufficient by itself for this issue if each
variant reloads the model. The matrix must either:

- use a resident runner that loads the model once per variant and executes the
  fixed ask set, or
- explicitly mark `resident_session_reused=false` and fail the resident-matrix
  acceptance.

## Required Fields

Each matrix variant should record:

- `variant_id`
- `requested_thread_count`
- `effective_thread_count`
- `thread_env`: `RAYON_NUM_THREADS`, `BITNET_CPU_THREADS`,
  `BITNET_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`
- `process_affinity_mask`
- `affinity_classification`
- `affinity_classification_status`
- `windows_power_scheme`
- `ac_battery_state`
- `thermal_availability`
- `temperature_c`, only when measured
- `cpu_utilization_per_logical_processor`, when available
- `frequency_or_throttle_proxy`, when available
- `resident_session_reused`
- `model_loaded_once`
- `tokenizer_loaded_once`
- `fallback_used`
- `claim_boundary`

Each profile sample should record:

- `profile_id`
- `case_id`
- `prompt_token_count`
- `generated_token_count`
- `tokenize_ms`
- `prefill_ms`
- `first_token_ms`
- `decode_total_ms`
- `detokenize_ms`
- `total_response_ms`
- `answer_gate_passed`
- `generated_token_ids_available`

Unavailable fields must be explicit nulls with a status string such as
`not_exposed`, not zero.

## Windows Telemetry Sources

Use these as the first telemetry sources and record failures verbatim in the
receipt:

```powershell
powercfg /getactivescheme
Get-CimInstance Win32_Battery
Get-CimInstance Win32_Processor
Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor
Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation
Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi
```

For process affinity, record the mask from the measurement process itself. If a
wrapper launches child processes, the wrapper must record the child PID and the
child process affinity after launch.

## Acceptance Mapping

The issue can close only when:

- current default, 1-thread, 4-thread, and 8-thread variants are present;
- every variant uses the same resident ask set;
- every variant records requested and effective thread count;
- every variant records power scheme and AC/battery state;
- thermal, utilization, and frequency fields are either measured or explicitly
  marked unavailable;
- fallback is false for every variant and profile;
- optional affinity variants include honest mask and classification evidence;
- the receipt says this is measurement context, not CPU speedup, default-tuning,
  route-policy, power, accelerator, or BitNet proof.
- the runner or receipt-builder surface has fail-closed fixture validation for
  missing required variants, missing fallback status, and non-resident samples.

## Claim Boundary

This plan does not add:

- new inference evidence;
- a CPU speedup claim;
- a route-policy change;
- default thread tuning;
- affinity pinning;
- low-power battery evidence;
- BitNet QK256/I2_S behavior evidence.

It only defines the dense Rust GGUF CPU thread/core matrix needed before those
decisions are safe.
