# Lunar Lake CPU Route Decision Memo

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1096](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1096), [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069), [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071)
Linked PRs: n/a
Support-tier impact: no promotion; review-only CPU route decision
Policy impact: no policy exception

## Recommendation

Keep Rust GGUF CPU as the dense SLM correctness and fallback baseline for now.
Do not start a Rust GGUF CPU optimization PR yet.

Evaluate OpenVINO CPU only as a separate dense SLM CPU candidate/control path
until a fair benchmark package exists. The current OpenVINO CPU evidence is
useful because corpus-v2 passes with fallback false and direct generated-token
visibility, but it is not a matched-format CPU speedup claim and it does not
replace the Rust GGUF CPU route.

This memo closes the decision gap from #1096. It does not change route policy,
run inference, refresh receipts, promote OpenVINO CPU, claim a speedup, claim a
power advantage, or prove BitNet QK256/I2_S behavior.

## Current Evidence

| Evidence | Current finding | Decision effect |
| --- | --- | --- |
| `lunar-lake-cpu-slow-path.md` | Rust GGUF CPU is slow after reload is removed; prefill, first-token, and decode remain large costs | Optimization needs phase and platform attribution before code changes |
| `lunar-lake-cpu-slm-runtime-comparison.json` | OpenVINO CPU corpus-v2 now passes, but `benchmark_qualified=false` | Use as route/context evidence, not speedup proof |
| `lunar-lake-openvino-token-visibility.md` | OpenVINO CPU has direct generated token IDs from the current corpus-v2 evidence | Token visibility is not the CPU comparison blocker |
| `lunar-lake-cpu-thread-core-matrix.md` | Dense Rust GGUF CPU lacks a thread/core matrix on 258V | Do not tune thread count or affinity defaults yet |
| #1069 | Resident CPU no-reload timing refresh remains open | Measurement subissue, not a route decision by itself |
| #1071 | Thread/core matrix evidence remains open | Measurement subissue, not a route decision by itself |

The refreshed runtime comparison records:

- Rust CPU route: `dense_slm_default_cpu`, `cpu-rust`,
  `resident_cpu_rust_gguf`, GGUF Q8_0, fallback false.
- OpenVINO CPU route: `dense_slm_openvino_cpu_candidate`,
  `openvino_genai`, `Intel(R) Core(TM) Ultra 7 258V`, OpenVINO IR
  INT4_SYM, fallback false.
- OpenVINO CPU corpus-v2: answer gate passed with direct generated token IDs.
- Benchmark qualification: false because model format, timing scope,
  prompt-render/tokenization visibility, and matched-profile coverage still do
  not align.

## Decision Table

| Option | Decision | Why | Next allowed PR |
| --- | --- | --- | --- |
| Optimize Rust GGUF CPU now | Defer | Current evidence names likely costs but not the exact target, thread/core behavior, or success metric | #1069 resident no-reload timing refresh or #1071 thread/core matrix |
| Evaluate OpenVINO CPU | Keep as separate candidate/control | OpenVINO CPU corpus-v2 passes, but GGUF Q8_0 and OpenVINO IR INT4_SYM are different runtime/model scopes | Matched-profile comparison schema or receipt refresh that keeps non-equivalence explicit |
| Keep CPU fallback/correctness baseline | Yes | Rust GGUF CPU is the known dense SLM local baseline and remains separate from accelerator proof | Docs/review closeout only unless a receipt invalidates it |
| Promote OpenVINO CPU for auto-route | Blocked | No promotion package proves exact-profile advantage under accepted CPU route scope | Route-policy PR only after fair-benchmark and product-scope gates pass |
| Treat OpenVINO CPU as Rust CPU speedup | Rejected | Model format and timing scope mismatch block engine-parity or matched-format speedup language | None; wording must stay fail-closed |

## Fair Benchmark Gate

A CPU comparison can become benchmark-qualified only when one package records
all of these fields for every compared profile:

| Gate | Requirement |
| --- | --- |
| Route identity | Requested and selected backend, runtime API, resolved device, route ID, proof family, and fallback status |
| Model scope | Exact source model, model format, quantization, tokenizer source, prompt template, and whether formats match |
| Prompt scope | Same corpus cases or benchmark prompts, rendered prompt identity, prompt token IDs or explicit unavailability, and generation config |
| Timing scope | Same cold/warm/resident mode, pipeline or model construction treatment, prompt render, tokenize, prefill, first token, decode, detokenize, quality gate, receipt write, and total response |
| Token visibility | Direct generated-token IDs for token-level comparison; retokenized or text-only evidence remains diagnostic |
| Profile coverage | Same named profile, prompt-token bounds, output-token bounds, and matched Rust/OpenVINO samples |
| Telemetry context | Windows power scheme, AC/battery state, thermal availability, thread count, affinity status, and frequency or throttle proxy when available |
| Claim boundary | Explicit statement of whether the comparison is engine parity, route/profile comparison, candidate context, or promotion evidence |

If model formats differ, the comparison may still support a route/profile
candidate review, but it must not claim matched-format engine parity. If timing
scopes differ, any ratio remains diagnostic until the receipt explains and
qualifies the difference.

## Block And Unblock Conditions

### Rust GGUF CPU Optimization

Blocked until at least one measurement issue identifies a target and success
metric:

- #1069 refreshes resident no-reload timing with per-prompt phase accounting;
- #1071 records default, 1-thread, 4-thread, and 8-thread dense Rust GGUF
  resident timing with power, scheduler, and telemetry context;
- a phase-attribution receipt names whether prefill, first token, decode,
  tokenization, prompt rendering, receipt overhead, or thread placement is the
  target.

Unblocked only for a narrow PR whose acceptance says exactly which metric moves
and which claim remains out of scope.

### OpenVINO CPU Candidate Evaluation

Allowed as a candidate/control path if the PR keeps these boundaries:

- OpenVINO CPU is not a replacement for Rust GGUF CPU;
- OpenVINO IR INT4_SYM and GGUF Q8_0 are not treated as matched engines;
- pipeline construction, generation wall time, tokenization, prompt rendering,
  and total response are separately named;
- direct generated-token visibility is recorded but not used to erase model
  format mismatch;
- route-profile comparison is separated from CPU speedup language.

Promotion remains blocked until the route-promotion package proves an accepted
exact-profile advantage and states whether the product accepts OpenVINO CPU as
a separate model/export route.

### CPU Fallback Status

Rust GGUF CPU remains the correctness/fallback baseline unless:

- its answer gates fail;
- fallback appears in strict CPU evidence;
- a shared tokenizer, sampler, model, or semantic change invalidates the
  current receipt set;
- another route earns an exact-profile promotion package for the same profile.

Measurement subissues do not change this status by themselves.

## Next Smallest PR

Do not start with CPU optimization.

The next small PR should be one of:

1. #1069 resident CPU no-reload timing refresh with stronger phase accounting.
2. #1071 thread/core matrix receipt for dense Rust GGUF resident asks.
3. A comparison-schema guard that keeps `benchmark_qualified=false` when model
   formats or timing scopes differ, while still allowing candidate context.

Each of those PRs should remain docs, receipt, schema, or validation scoped.
None should change route policy unless a later review links a completed
promotion package.

## Claim Boundary

This memo does not add:

- new Lunar Lake inference;
- route-policy mutation;
- CPU optimization;
- OpenVINO CPU promotion;
- CPU speedup or matched-engine speedup claims;
- power-advantage evidence;
- low-power evidence;
- generated dashboards;
- native OpenCL or NPU proof;
- BitNet QK256/I2_S behavior proof.

It only chooses the current CPU route posture and defines the evidence required
before optimization, OpenVINO CPU promotion, or CPU speedup language is safe.
