# Lunar Lake CPU Route Decision Memo

Status: review
Owner: intel/openvino
Created: 2026-05-31
Post-matrix refresh: 2026-06-01
Post-source-run refresh: 2026-06-02
Post-field rerun refresh: 2026-06-02
Post-scope-contract refresh: 2026-06-02
Post-reviewability-contract refresh: 2026-06-02
Post-physical-package refresh: 2026-06-02
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1122](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1122), [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069), [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071), [#1186](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186), [#1195](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1195), [#1201](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1201), [#1209](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1209), [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232), [#1277](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1277), [#1280](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1280), [#1281](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1281), [#1291](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1291), [#1311](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1311), [#1365](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1365)
Linked PRs: [#1132](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1132), [#1156](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1156), [#1182](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1182), [#1194](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1194), [#1207](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1207), [#1208](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1208), [#1255](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1255), [#1266](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1266), [#1279](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1279), [#1283](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1283), [#1290](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1290), [#1292](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1292), [#1319](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1319), [#1334](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1334)
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

This memo records the decision from #1122, landed by #1132. #1156 later added
the fail-closed comparison guard for the non-equivalence boundary described
here: CPU comparison receipts must keep benchmark qualification false when
model formats or timing scopes differ, and the qualification fields must agree.
Issue #1182 then documented and guarded the `lunar-lake cpu-slm-resident-session`
command surface as a no-new-inference summarizer/validator over existing
resident-session receipts. #1194 added the
`lunar-lake cpu-slm-thread-core-matrix` receipt-builder and validator contract
for default, 1-thread, 4-thread, and 8-thread variants. #1207 closed the
source-receipt enrichment gap, and #1208 closed #1071 with physical matrix
evidence. The matrix is now evidence for review, not a CPU tuning unlock.
This memo does not change route policy, run inference, refresh receipts,
promote OpenVINO CPU, claim a speedup, claim a power advantage, or prove
BitNet QK256/I2_S behavior.
PR #1255 later added a resident CPU `measurement_qualification` guard to make
the same decision visible in the current receipt: resident no-reload context
remains useful, but resident phase evidence is not benchmark-qualified.
PR #1279 then added the resident-specific source fixture for the physical CPU
package, and #1280 is now closed by #1334. The committed package records 33
prompts and 32 warm asks after the first resident ask with fallback false,
model/tokenizer loaded once, quality passing, and determinism passing. #1281
then closed through #1290, which added prompt-render
timing, quality-gate timing, detokenization summary exposure, and clearly
defined resident memory lifecycle samples. #1334 shows those fields in the
committed summary and narrows the remaining resident strict blockers to
`receipt_write_ms` and `telemetry_ms`. #1291 is closed by #1292: current
resident summaries must keep
profile `receipt_write_ms` and `telemetry_ms` as explicit `not_exposed` fields
instead of backfilling them from aggregate/session observations unless a later
spec or research contract defines the source, scope, and summarizer rule.
Issue #1311 is closed by #1319. The accepted status contract adds a separate
diagnostic-reviewability path when the only remaining blockers are the
profile `receipt_write_ms` and `telemetry_ms` contract-not-exposed fields from
the #1291/#1292 scope contract, while preserving strict
`resident_phase_qualified=false` and `benchmark_qualified=false`.
PR #1283 recorded the original boundary in the CPU slow-path research note,
and #1319 made the reviewable-versus-qualified split visible in receipts.
PR #1334 applies that split to the physical resident package:
`diagnostic_package_reviewable=true`, `resident_phase_qualified=false`, and
`benchmark_qualified=false`.
Issue #1365 now owns the matched Rust GGUF CPU versus OpenVINO CPU comparison
contract. It keeps OpenVINO CPU as a diagnostic candidate/control while the
next comparison package names model-format alignment, timing-scope alignment,
direct-token visibility, fallback status, telemetry context, and fail-closed
benchmark qualification.

## Current Evidence

| Evidence | Current finding | Decision effect |
| --- | --- | --- |
| `lunar-lake-cpu-slow-path.md` | Rust GGUF CPU is slow after reload is removed; prefill, first-token, and decode remain large costs | Optimization needs phase and platform attribution before code changes |
| `lunar-lake-cpu-slm-runtime-comparison.json` | OpenVINO CPU corpus-v2 now passes, but `benchmark_qualified=false`; #1156 guards this status when model formats or timing scopes differ | Use as route/context evidence, not speedup proof |
| `lunar-lake-openvino-token-visibility.md` | OpenVINO CPU has direct generated token IDs from the current corpus-v2 evidence | Token visibility is not the CPU comparison blocker |
| `lunar-lake-cpu-thread-core-matrix.md` | #1208 records the physical default / 1-thread / 4-thread / 8-thread matrix with `matrix_ready=true` and no gaps; default and `threads_1` both resolve to one effective thread, while 4-thread and 8-thread variants are slower in this run | Do not tune thread count or affinity defaults from this evidence |
| #1069 / #1182 | #1069 is closed as a resident-session command-surface review; #1182 did not add a fresh physical resident no-reload measurement source | Historical command-surface closeout, not a route decision or optimization unlock |
| #1071 / #1208 | Thread/core matrix evidence is closed and committed | Measurement evidence, not a route decision by itself |
| #1186 / #1194 | #1186 is closed by the no-inference thread/core matrix builder and validator in #1194 | Receipt-builder closeout, not physical matrix evidence |
| #1201 / #1207 | Source-receipt contract for the physical matrix is closed | Source-enrichment support, not an open blocker |
| #1209 | Post-matrix CPU review is closed | The review consumes #1208 and leaves only measurement-first follow-ups; do not optimize blindly |
| #1232 / #1255 | Resident Rust GGUF phase evidence successor remains open; #1255 added the original `measurement_qualification` fail-closed guard | #1334 now supplies the physical resident package and diagnostic-reviewable status, but strict `resident_phase_qualified=false` and `benchmark_qualified=false` remain |
| #1365 | Matched Rust GGUF CPU versus OpenVINO CPU comparison issue is open | Owns future CPU comparison package shape; does not promote OpenVINO CPU or claim speedup while model-format and timing-scope gates remain unresolved |
| #1277 / #1279 | Resident source-shape successor is closed by the committed `ci/quality/lunar-lake-resident-qwen25-cpu.yaml` fixture | Source fixture yields 33 prompts and 32 warm asks after first; it is not physical evidence by itself |
| #1280 / #1281 / #1291 / #1292 / #1311 / #1319 / #1334 | #1334 closes #1280 with the physical resident package; #1290 made prompt-render, quality-gate, detokenize, and memory lifecycle fields measurable; #1292 closed the receipt-write and telemetry scope contract by keeping those profile fields explicit `not_exposed`; #1319 closed #1311 by adding separate diagnostic reviewability while preserving strict false qualification | Treat the package as diagnostic resident CPU evidence for #1232. Do not optimize CPU, change route policy, promote OpenVINO CPU, claim benchmark qualification, or repeat #1280 artifacts unless a later issue defines a narrower missing evidence target |

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
| Optimize Rust GGUF CPU now | Defer | Current evidence names likely costs, and #1208 argues against blind thread-count tuning, but it still does not identify a safe runtime target or success metric | A narrow phase-attribution, resident no-reload, #1365 matched-comparison, or topology receipt only after the issue defines the metric |
| Evaluate OpenVINO CPU | Keep as separate candidate/control | OpenVINO CPU corpus-v2 passes, but GGUF Q8_0 and OpenVINO IR INT4_SYM are different runtime/model scopes | #1365 matched-profile comparison schema or receipt refresh that keeps non-equivalence explicit |
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

The #1156 guard makes this boundary executable for current CPU comparison
receipts. A receipt cannot mark the comparison benchmark-qualified while model
formats differ, while timing scopes differ, or while
`benchmark_qualification.qualified` and
`timing_scope_alignment.benchmark_qualified` disagree. That guard preserves
candidate context; it does not promote OpenVINO CPU or close the resident
timing question. The thread/core physical matrix is now closed by #1208, but
its result is a negative tuning signal, not an optimization unlock.

## Block And Unblock Conditions

### Rust GGUF CPU Optimization

Blocked until at least one measurement issue identifies a target and success
metric:

- #1334 records one model load, one tokenizer load, one separated first
  resident ask, and 32 additional warm asks as diagnostic resident evidence,
  while preserving strict false qualification;
- #1208 records default, 1-thread, 4-thread, and 8-thread dense Rust GGUF
  resident timing with power, scheduler, and telemetry context but does not
  show a thread-count win;
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

The remaining next small PRs are evidence work only:

1. Use [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232)
   to decide the next resident phase-evidence follow-up after the committed
   #1280/#1334 package. A follow-up may define scoped aggregate
   receipt-write/telemetry fields or a topology receipt, but it must keep
   strict resident qualification and benchmark qualification false unless a
   later contract revises the rule.
2. Use [#1365](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1365)
   for a matched Rust GGUF CPU versus OpenVINO CPU comparison refresh only
   after it can keep model-format, timing-scope, prompt-render, tokenization,
   and benchmark-qualification blockers explicit.
3. A later affinity/topology receipt only if P-core/E-core placement,
   frequency, utilization, or thermal context can be exposed accurately enough
   to make the result reviewable.

The comparison-schema guard that keeps `benchmark_qualified=false` when model
formats or timing scopes differ landed in #1156. Future CPU comparison work
should refresh matched-profile evidence, use completed measurement packages, or
harden a newly exposed gap, not repeat that guard.

Issue #1209 is closed as the post-matrix review. Issue #1232 remains the parent
resident phase evidence contract, while #1280 is closed by #1334 as the
physical source package. The #1291 issue is closed by #1292 as the accepted
receipt-write/telemetry scope boundary, and #1311 is closed by #1319 as the
reviewable-versus-qualified status decision before any runtime optimization or
route-policy PR.

Those PRs should remain docs, receipt, schema, or validation scoped.
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
