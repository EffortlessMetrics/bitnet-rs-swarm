# Kaby Lake SLM CPU Performance Dashboard

This dashboard is the baseline for i5-8250U dense SLM performance work. It
summarizes existing strict Qwen3-0.6B Q8_0 receipts only; it is not a sustained
throughput claim and it does not broaden support to Q4/Q5, server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256.

## Evidence Set

| Evidence | Path | Role |
| --- | --- | --- |
| Thread envelope | `ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-thread-timing-envelope.json` | 1, 2, 4, and 8 thread warm-session timing comparison |
| Thread validation | `ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-thread-timing-envelope-validation.json` | Validates strict provenance, quality, determinism, and no fallback across thread counts |
| Operator profile | `ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-operator-profile.json` | Default operator evidence with process memory, storage/free-space, warm-session timing, and unsupported-path fields |
| Greedy sampler fast path | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path-validation.json` | Validates that the guarded greedy no-penalty sampler fast path preserves the 4-thread generated IDs/text while sampler decode allocations remain zero |
| Logits extraction isolation | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse-validation.json` | Validates that direct tensor argmax bypasses full logits Vec extraction only where the sampler fast path is exact, while preserving generated IDs/text |
| Repetition-penalty logits reuse | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse-validation.json` | Validates that default repetition-penalty decode steps reuse a host logits scratch buffer instead of allocating fresh logits vectors, while preserving generated IDs/text |
| Warm-session sampler reuse | `ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-kv-temp-reuse-validation.json` | Validates that the temperature-zero warm-session profile reuses one sampler across prompts while preserving generated IDs/text and strict provenance |
| KV cache session reuse | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-kv-cache-session-reuse.json` | Records one CPU KV cache reused across prompts and cleared per prompt for prompt isolation |
| Prompt token cache | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json` | Validates that repeated rendered prompts reuse token IDs while preserving generated IDs/text and strict provenance |
| Prefill allocation attribution | `ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prefill-attribution-validation.json` | Validates that prompt prefill attribution preserves behavior and identifies `prompt_prefill.forward` as the dominant remaining allocation boundary |
| Packed Q8_0 sidecar runtime proof gate | `ci/slm-cpu/intel-i5-8250u/2026-05-19/qwen3-packed-q8-sidecar-runtime-proof-validation.json` | Records that packed Q8_0 sidecar runtime execution remains blocked because production dispatch still preserves eager F32 and no after-execution receipts exist |
| Post-bridge packed-Q8 counter classification | `ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-077-post-bridge-counter-classification.json` | Records real i5-8250U sidecar instrumentation counters and identifies `packed_matvec_compute` as the dominant exact-tensor sidecar cost |
| Post-aligned packed-Q8 matvec classification | `ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-079-post-aligned-matvec-classification.json` | Preserves the Qwen3 behavior oracle while recording a bounded counter-level `packed_matvec_ns` reduction against the SLM-CPU-077 sidecar oracle |
| Logits/output-head boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-23/qwen3-slm-cpu-091-logits-output-boundary.json` | Classifies the remaining `model.logits` / output-head boundary as owned Candle Tensor output plus optional host extraction, without claiming a runtime optimization |
| Runtime output tensor storage gate | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-096-runtime-output-tensor-storage-gate.json` | Records that inner packed-Q8 matvec helpers can fill caller-owned slices, but full runtime Tensor output reuse remains blocked by public Candle owned-storage construction |
| Fused output consumer boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-098-fused-output-consumer-boundary.json` | Classifies the exact `attention.q_proj` fused-consumer boundary as blocked by downstream Candle Tensor consumers unless a typed fused Q projection consumer owns reshape, q_norm, RoPE, trace/workspace identity, and attention-head handoff semantics |
| Typed fused Q consumer contract | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-099-typed-fused-q-consumer-contract.json` | Defines the design-only API and receipt-safety contract for a future exact `attention.q_proj` fused consumer while keeping runtime execution, allocation claims, and speed claims disabled |
| Typed fused Q consumer implementation gate | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-100-typed-fused-q-consumer-implementation-gate.json` | Records that the exact `attention.q_proj` fused consumer remains runtime-disabled until a typed attention-head buffer/view owns reshape, q_norm, RoPE, trace identity, and score handoff semantics |
| Typed attention-head view gate | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-101-typed-attention-head-view-gate.json` | Defines the runtime-disabled typed Q-head view contract and records the remaining q_norm, RoPE, trace identity, score-handoff, and receipt-safety blockers |
| Typed attention-head consumer gate | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-102-typed-attention-head-consumer-gate.json` | Classifies the current typed Q-head consumer boundary as blocked before q_norm/RoPE/trace/score handoff, records candidate materialization points, and keeps runtime execution, allocation claims, and timing claims disabled |
| Typed q_norm/RoPE consumer gate | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-103-typed-qnorm-rope-consumer-gate.json` | Records the next typed q_norm/RoPE consumer boundary as blocked at `typed_q_norm_consumer`, names the exact Tensor APIs and receipt-safety gaps, and keeps runtime execution, allocation claims, and timing claims disabled |
| q_norm materialization boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-104-qnorm-materialization-boundary.json` | Selects `q_norm_input_candle_tensor_boundary` as the only accepted materialization boundary for the next proof slice, preserving existing Candle q_norm, RoPE, trace, and score consumers while keeping runtime execution and allocation/timing claims disabled |
| q_norm input proof gate | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-105-qnorm-input-proof-gate.json` | Blocks proof of the selected `q_norm_input_candle_tensor_boundary` until a runtime-disabled hook, Qwen3 and Qwen2.5 before/after strict CPU receipt pairs, a fail-closed comparator, q_norm input tensor identity, and accumulator-order evidence exist |
| q_norm input receipt comparator | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-106-qnorm-input-receipt-comparator.json` | Defines the fail-closed before/after receipt identity comparator for the selected `q_norm_input_candle_tensor_boundary`, burning down the comparator blocker while keeping runtime execution, proof readiness, and allocation/timing claims disabled |
| q_norm input runtime-disabled hook gate | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-107-qnorm-input-runtime-hook-gate.json` | Defines the runtime-disabled hook identity and q_norm-input tensor-identity receipt surface for the selected boundary while keeping proof blocked on Qwen3/Qwen2.5 before/after receipts and accumulator-order evidence |
| Qwen3 q_norm input receipt-pair blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-108-qnorm-input-qwen3-receipt-pair-blocker.json` | Verifies the Qwen3 Q8_0 model is present but blocks receipt-pair collection because warm-session receipts do not yet emit `dense_q8_hook.q_norm_input_tensor_identity` |
| Qwen3 q_norm input receipt pair | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-110-qnorm-receipt-pair-validation.json` | Collects the Qwen3 Q8_0 before/after strict CPU warm-session receipt pair after the identity field landed; generated IDs and decoded text match, but tensor fingerprint capture and Qwen2.5 coverage remain blocked, so no allocation, timing, throughput, or default-runtime claim is made |
| Qwen3 q_norm tensor fingerprint blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-111-qnorm-tensor-fingerprint-blocker.json` | Precisely blocks f32-le q_norm-input tensor fingerprint capture because the warm-session receipt boundary receives metadata but not the materialized Candle Tensor or a receipt-safe host f32-le slice |
| Qwen3 q_norm fingerprint diagnostic capture | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-112-qnorm-fingerprint-diagnostic-capture.json` | Adds an opt-in trace-only `attention.q_norm_input` fingerprint record for the exact layer-0 Qwen3 Q8_0 q_proj boundary, recording shape, dtype, source tensor, boundary, and f32-le SHA256 without tensor contents while preserving eager F32 as the default runtime |
| Qwen3 q_norm fingerprint artifact | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-113-qnorm-fingerprint-artifact.json` | Captures a real i5-8250U Qwen3 Q8_0 `attention.q_norm_input` f32-le SHA256 artifact with strict GGUF tokenizer authority, `cpu-rust`, `fallback_used=false`, and no tensor contents or performance claim |
| q_norm fingerprint receipt-pair review | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-114-qnorm-fingerprint-receipt-pair.json` | Compares the Qwen3 q_norm fingerprint artifact against the existing before/after receipt-pair evidence and records the exact Qwen2.5 blocker: Qwen3 has a real q_norm fingerprint, but the before/after warm-session receipts do not carry f32-le tensor fingerprints, and the accepted Qwen2.5 Q8_0 artifact has no q_norm-input stage to fingerprint |
| q_norm proof next boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-115-qnorm-proof-next-boundary.json` | Resolves the Qwen3-only q_norm-input blocker by selecting the shared `attention.q_proj_output_pre_optional_qnorm` boundary that exists before Qwen3's optional q_norm and also exists on Qwen2.5 Q8_0 |
| shared q_proj output hook gate | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-116-qproj-output-pre-qnorm-hook-gate.json` | Defines the fail-closed receipt/comparator contract for the shared pre-q_norm Q projection output boundary and blocks runtime evidence until an implementation-capable slice adds the diagnostic hook |
| shared q_proj output hook surface | `ci/slm-cpu/intel-i5-8250u/2026-05-25/qwen3-slm-cpu-117-qproj-output-pre-qnorm-hook.json` | Adds the opt-in runtime-disabled `attention.q_proj_output_pre_optional_qnorm` fingerprint hook and fail-closed comparator artifact kind without promoting packed_q8_sidecar or claiming behavior, allocation, timing, or throughput |
| q_proj output receipt-pair blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-119-qproj-output-receipt-pair-blocker.json` | Captures real Qwen3 and Qwen2.5 hook evidence, restores the accepted Qwen2.5 artifact, and blocks behavior proof because Qwen3 sidecar-gated after-run fails before receipt emission while Qwen2.5 preserves generated output but changes the q_proj-output f32 fingerprint |
| q_proj sidecar transpose guard | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-120-sidecar-transpose-guard.json` | Adds a fail-closed runtime guard for packed Q8_0 sidecar payloads whose GGUF byte order has only been shape-reshaped to Candle runtime matrix orientation; Qwen3 still fails before a post-guard receipt, so no allocation, timing, default-runtime, or packed-Q8 behavior claim is made |
| Bounded KV cache pre-boundary allocation | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-slm-cpu-121-bounded-kv-cache.json` | Resolves the Qwen3 post-guard pre-boundary full-context KV allocation failure by allocating prompt-plus-generation bounded KV capacity; the run reaches receipt emission and the layer-0 `attention.q_proj_output_pre_optional_qnorm` fingerprint, without claiming q_proj sidecar behavior, timing improvement, allocation-performance improvement, or default-runtime promotion |

Qwen3 rows use:

```text
model = Qwen/Qwen3-0.6B-GGUF / Qwen3-0.6B-Q8_0.gguf
sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
backend = cpu-rust
tokenizer.source = gguf_metadata
tokenizer.strict = true
fallback_used = false
prompt_template = qwen
qwen_no_think = true
temperature = 0.0
greedy = true
```

## Dashboard Refresh State

This refresh is current through the SLM-CPU-121 bounded KV-cache pre-boundary
allocation slice. It records that Qwen3 Q8_0 strict CPU generation now reaches
post-guard receipt emission and the layer-0
`attention.q_proj_output_pre_optional_qnorm` fingerprint after allocating only
the prompt-plus-generation KV capacity required by the tiny proof run. That is a
blocker resolution for the SLM-CPU-120 167772160-byte full-context KV allocation
failure, not a q_proj sidecar behavior proof, packed-Q8 default-runtime
promotion, allocation-performance claim, timing claim, sustained-throughput
claim, or answer-quality claim.

Earlier context through SLM-CPU-120: the shared q_proj-output sidecar
transpose-order guard. It records the opt-in trace-only f32-le tensor fingerprint surface for
the selected Qwen3 q_norm-input boundary and a real i5-8250U capture, then keeps
that boundary fail-closed because the before/after warm-session receipts still do
not carry f32-le tensor fingerprints and the accepted Qwen2.5 Q8_0 artifact has
no q_norm-input stage to fingerprint. SLM-CPU-115 therefore selects the shared
`attention.q_proj_output_pre_optional_qnorm` boundary as the next evidence target
for Qwen3 Q8_0 and Qwen2.5 Q8_0 rather than overstating q_norm coverage.
SLM-CPU-116 defines the fail-closed receipt/comparator contract for that shared
boundary, SLM-CPU-117 adds the opt-in runtime-disabled diagnostic hook and
comparator artifact surface, SLM-CPU-118 blocks behavior proof until exact
Qwen3 and Qwen2.5 before/after receipt pairs carry that hook fingerprint, and
SLM-CPU-119 restores the accepted Qwen2.5 Q8_0 artifact and captures the next
real evidence. SLM-CPU-120 adds a fail-closed guard so packed Q8_0 sidecar
runtime execution is declined when the GGUF payload byte order has only been
shape-reshaped into the Candle runtime matrix shape without a proven payload
reorder. It also records that the real Qwen3 Q8_0 post-guard run still fails
before a receipt or layer-0 q_proj-output fingerprint with the same 167772160
byte allocation failure, so that remaining blocker is now classified as
pre-boundary rather than accepted as q_proj sidecar behavior evidence. It
does not add a runtime optimization. It
re-indexes the merged Kaby Lake Qwen3 Q8_0 evidence after KV-cache reuse,
prompt-token caching, prefill attribution, the post-aligned exact-tensor
packed-Q8 matvec artifact, the residual-add output-storage blocker, the
logits/output-head boundary, the packed-Q8 caller-output-slice helper gate, and
the typed fused Q projection consumer, attention-head view, attention-head
consumer, q_norm/RoPE consumer blockers, and the selected q_norm-input
materialization boundary plus its proof blocker, comparator contract,
runtime-disabled hook/tensor-identity surface, fingerprint-only diagnostic trace,
receipt-pair review blocker, cross-family next-boundary selection, and shared
q_proj-output hook gate.

The current operator default remains evidence-scoped to the recorded 4-thread
operator profile. The default production runtime remains `eager_f32_candle`.
The packed-Q8 sidecar path remains opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight` / `blk.0.attn_q.weight`.

The current performance targets are:

1. Allocation/layout work around `prompt_prefill.forward` and `model.forward`,
   anchored by the SLM-CPU-035 prefill attribution receipt.
2. Exact-tensor packed-Q8 matvec compute work, anchored by the SLM-CPU-077
   counter pack, SLM-CPU-079 post-aligned counter classification, and
   SLM-CPU-096's runtime Tensor storage blocker.

Both targets remain gated by the Qwen3 Q8_0 behavior oracle: model SHA, strict
GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU
backend/kernel identity, dense hook identity where applicable, and
`fallback_used=false`.

SLM-CPU-097 queued the fused output consumer gate after SLM-CPU-096. SLM-CPU-098
classified the concrete fused-consumer boundary: the exact packed-Q8 sidecar
matvec can fill caller-provided slices, but `layers.0.attention.q_proj.weight`
immediately feeds Tensor-shaped consumers for reshape, transpose, optional q_norm,
RoPE, trace/workspace identity, and attention-head handoff. A future improvement
must design a typed fused Q projection consumer for those semantics before
claiming allocation or timing improvement, and remains before/after receipt gated
by the same Qwen3 Q8_0 behavior oracle.

SLM-CPU-099 defines that contract as a machine-checkable, design-only surface.
The exact tensor remains `layers.0.attention.q_proj.weight`, and the contract
requires a future fused consumer to own the packed-Q8 matvec output slice,
q-projection reshape, transpose, optional q_norm, RoPE, trace/workspace identity,
and attention-head handoff. Runtime fused consumer execution remains disabled,
intermediate returned Candle Tensor avoidance is not active, and allocation or
speed claims remain blocked until repeated before/after Qwen3 Q8_0 receipts
preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs,
decoded text, backend/kernel identity, dense hook identity, and
`fallback_used=false`.

SLM-CPU-100 attempted that implementation gate and kept runtime execution
disabled. The inner packed-Q8 matvec can fill caller-owned slices, but the next
safe work is a typed attention-head buffer/view that carries the Q projection
through reshape, transpose, optional q_norm, RoPE, trace/workspace identity, and
attention score handoff without constructing an intermediate returned Candle
`Tensor`. The recorded blockers are `q_heads_tensor_semantics`,
`q_norm_tensor_api`, `rope_tensor_api`, `trace_workspace_tensor_identity`,
`attention_handoff_tensor_contract`, and `receipt_safety_evidence`. No
allocation reduction, timing improvement, or default-runtime change is claimed.

SLM-CPU-101 defined that typed attention-head view as a runtime-disabled
contract. SLM-CPU-102 then classifies the consumer side of the same boundary:
projection-slice ingress and the logical Q-head view are representable without a
returned Candle `Tensor`, but q_norm, RoPE, trace/workspace identity, and
attention score handoff still require Tensor-backed APIs or a separately proven
single-materialization boundary. The current first blocking stage is
`q_norm_consumer`; no materialization point is accepted as behavior-preserving
without before/after Qwen3 Q8_0 and Qwen2.5 Q8_0 CPU receipts. Runtime fused Q
execution remains disabled, and no allocation, timing, sustained-throughput,
Q4/Q5, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 claim is
made.

SLM-CPU-103 records the typed q_norm/RoPE consumer gate as a
machine-checkable blocker. The logical Q-head view can be represented without a
returned Candle `Tensor`, but the typed q_norm/RoPE consumer still stops at
`typed_q_norm_consumer`: the current runtime uses
`candle_nn::LayerNorm::forward(&Tensor)` for q_norm,
`RotaryEmbedding::apply(&Tensor, position)` for RoPE, Tensor-backed trace
identity, and Tensor-backed attention-score handoff. No single materialization
point is accepted yet. The next safe slice is either a behavior-equivalent typed
q_norm/RoPE kernel pair or exactly one proven materialization boundary before
attention scores, with Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after receipts before
any allocation or timing claim.

SLM-CPU-104 selects exactly one of those materialization candidates:
`q_norm_input_candle_tensor_boundary`. This is intentionally conservative. It
keeps q_norm on `candle_nn::LayerNorm::forward(&Tensor)`, RoPE on
`RotaryEmbedding::apply(&Tensor, position)`, trace identity on
`TransformerAttentionOutputSourceTensors`, and score handoff on the existing
Tensor-backed `prepare_attention_scores` path. Runtime fused Q execution remains
disabled, the default runtime remains `eager_f32_candle`, and the boundary is
not an allocation or timing claim. A future runtime-adjacent slice must prove the
selected boundary with before/after Qwen3 Q8_0 and Qwen2.5 Q8_0 receipts before
claiming any improvement.

SLM-CPU-105 kept that boundary blocked rather than treating the SLM-CPU-104
selection as proof. The missing surfaces were explicit: no runtime-disabled hook
currently labels a q_norm-input materialization candidate separately from
`eager_f32_candle`, no Qwen3 Q8_0 or Qwen2.5 Q8_0 before/after strict CPU
receipt pairs exist for the selected boundary, the comparator surface was not
yet defined, no receipt records q_norm-input tensor identity, and
accumulator-order equivalence remains unproven.

SLM-CPU-106 burns down only the comparator blocker. The new comparator contract
requires model SHA, GGUF tokenizer authority, strict tokenizer mode, prompt-ID
digest, generated-ID digest, decoded-text digest, selected CPU backend/kernel
identity, dense hook identity, the `q_norm_input_candle_tensor_boundary` label,
q_norm-input tensor identity, and `fallback_used=false`, and it fails closed on
missing fields, mismatches, fallback, non-CPU backend identity, non-strict
tokenizer mode, or the wrong q_norm-input boundary. Runtime execution remains
disabled, the default runtime remains `eager_f32_candle`, and this is not an
allocation, timing, throughput, Q4/Q5, server, accelerator, Qwen3.5, or BitNet
QK256 claim.

SLM-CPU-107 burns down the runtime-hook and tensor-identity-surface blockers
without proving the boundary. The hook identity is
`layers.0.attention.q_proj.weight:q_norm_input_candle_tensor_boundary:runtime_disabled`
and the after-receipt field is `dense_q8_hook.q_norm_input_tensor_identity`.
That surface must carry the boundary label, source tensor, source stage, shape,
dtype, dense-hook identity, and f32-le tensor fingerprint. Packed-Q8 sidecar
execution remains disabled, the default runtime remains `eager_f32_candle`, and
proof still requires Qwen3 Q8_0 plus Qwen2.5 Q8_0 before/after strict CPU
receipts that pass the comparator before any allocation or timing claim.

SLM-CPU-108 verified the local Qwen3 Q8_0 model SHA but did not collect an
incomplete receipt pair because the warm-session receipt writer still lacked
`dense_q8_hook.q_norm_input_tensor_identity`.

SLM-CPU-109 added that missing warm-session receipt field while keeping the
receipt fail-closed: `proof_ready=false`, the q_norm-input tensor fingerprint is
explicitly unavailable, the default runtime remains `eager_f32_candle`, and the
packed-Q8 sidecar remains disabled by default.

SLM-CPU-110 collects the first Qwen3 Q8_0 before/after strict CPU receipt pair
after that field landed. The before receipt uses `eager_f32_candle`; the after
receipt selects the exact `layers.0.attention.q_proj.weight`
`packed_q8_sidecar` candidate. Both receipts preserve the model SHA, strict
GGUF tokenizer authority, CPU backend, `fallback_used=false`, prompt IDs,
generated IDs, decoded text, and the q_norm-input identity object. This is a
behavior-preservation receipt pair for Qwen3 only. It is not a full
materialization-boundary proof because the f32-le q_norm-input tensor
fingerprint is still not captured, Qwen2.5 coverage is still missing, and
accumulator-order equivalence remains unproven. It makes no allocation
reduction, timing improvement, sustained-throughput, default-runtime, Q4/Q5,
server, accelerator, Qwen3.5, or BitNet QK256 claim.

SLM-CPU-115 resolves the SLM-CPU-114 proof blocker into a safer cross-family
boundary instead of treating a Qwen3-only q_norm stage as Qwen-family proof. The
selected next evidence target is `attention.q_proj_output_pre_optional_qnorm`:
the dense Q projection output after `layers.0.attention.q_proj.weight` /
`blk.0.attn_q.weight` and before optional q_norm/k_norm, RoPE,
trace/workspace identity, or attention-score handoff. This boundary exists on
both Qwen3 Q8_0 and the accepted Qwen2.5 Q8_0 artifact, while the previous
`q_norm_input_candle_tensor_boundary` remains Qwen3-specific under the current
artifacts. Runtime execution remains unchanged, the default runtime remains
`eager_f32_candle`, and before any packed-Q8 promotion or allocation/timing claim
the new boundary still requires Qwen3 and Qwen2.5 before/after receipts with
matching model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text,
CPU backend/kernel identity, dense hook identity, f32-le tensor fingerprints, and
`fallback_used=false`.

SLM-CPU-116 keeps that selected shared boundary fail-closed. The current trace
surface can fingerprint Qwen3 `attention.q_norm_input`, but it cannot yet expose
`attention.q_proj_output_pre_optional_qnorm` for both Qwen3 and Qwen2.5. The new
gate defines the required runtime-disabled hook identity, strict receipt fields,
and comparator fail conditions, then records the exact blockers:
`runtime_disabled_trace_hook_missing`, `warm_session_receipt_field_missing`,
`comparator_boundary_variant_missing`, and `qwen25_artifact_pair_missing`.
Runtime execution remains unchanged, `eager_f32_candle` remains the default, and
no allocation, timing, throughput, Q4/Q5, server, accelerator, Qwen3.5, or BitNet
QK256 claim is made.

SLM-CPU-117 adds the missing opt-in trace surface for the shared boundary. When
Qwen tracing is explicitly active, the transformer emits an
`attention.q_proj_output_pre_optional_qnorm` f32-le fingerprint with source
tensor, GGUF tensor, boundary, dense-hook identity, shape, dtype, and
`runtime_disabled=true`. The reference comparator also accepts a fail-closed
`slm_qproj_output_pre_qnorm_hook_compare` artifact kind that requires the dense
hook identity and fingerprint. This is still only a diagnostic/proof surface:
before/after real Qwen3 and Qwen2.5 receipt pairs are still required before any
behavior, allocation, timing, default-runtime, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 claim.

SLM-CPU-119 captures the next proof slice in
`ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-119-qproj-output-receipt-pair-blocker.json`.
Qwen3 Q8_0 now has a before-side `attention.q_proj_output_pre_optional_qnorm`
fingerprint, but the sidecar-gated after-run fails before writing a receipt with
`memory allocation of 167772160 bytes failed`. Qwen2.5 Q8_0 has before/after
strict CPU receipts with identical model SHA, tokenizer authority, prompt IDs,
generated ID `19`, decoded text `4`, selected backend/kernel, and
`fallback_used=false`, but the q_proj-output f32 fingerprint changes from
`4cf085e0c5715156b96c59668a9601da8e5925fb96f2772ff4863cb67eade344` to
`4d69d3d003359fa0b221affec60e25f1c22b6146d52e91f69b4c7437c648b9ff`.
The boundary therefore remains not behavior-proven, and SLM-CPU-120 must burn
down those two concrete blockers before allocation, timing, default-runtime, or
throughput work can use this boundary as proof.

## Thread Envelope

The thread envelope is a bounded single-run comparison. Generated IDs are stable
across the tested thread counts, and all tested runs preserve `fallback=false`.

| Threads | Total session ms | Warm prompt wall ms | Prefill ms | Decode total ms | First token mean ms | Steady decode mean tok/s | Warm generated tok/s | Cold generated tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 137179.478 | 94609.006 | 65837.089 | 23650.743 | 12311.833 | 1.931 | 0.486 | 0.335 |
| 2 | 136826.761 | 95562.894 | 65459.825 | 23672.087 | 12405.667 | 1.898 | 0.481 | 0.336 |
| 4 | 135679.052 | 94416.265 | 64733.698 | 23725.190 | 12328.333 | 1.963 | 0.487 | 0.339 |
| 8 | 136190.928 | 94979.978 | 65590.721 | 24318.391 | 12332.833 | 1.907 | 0.484 | 0.338 |

The best recorded total session and steady decode values in this bounded set are
the 4-thread run. The envelope still records `operator_default_recommendation =
null` because thermal, power, and resident memory were not sampled in the thread
envelope itself.

## Operator Profile Baseline

The operator profile fills the host-context gap for the selected operator shape.
It uses 4 threads and records process memory and storage/free-space context.

| Field | Value |
| --- | --- |
| Threads | 4 |
| Model loaded once | true |
| Tokenizer loaded once | true |
| Prompt count | 6 |
| Prompt tokens | 176 |
| Generated tokens | 46 |
| Total session ms | 141643.848 |
| Model load ms | 37531.292 |
| Tokenizer load ms | 780.975 |
| Warm prompt wall ms | 98140.650 |
| Prefill ms | 66958.577 |
| Decode total ms | 26077.955 |
| First token mean ms | 12709.333 |
| First token p95 ms | 13931.000 |
| Decode generated tok/s | 1.764 |
| Warm prompt generated tok/s | 0.469 |
| Cold session generated tok/s | 0.325 |
| Resident memory bytes | 3045429248 |
| Virtual memory bytes | 3036508160 |
| Model path free bytes | 45233147904 |
| Receipt path free bytes | 45233147904 |
| Thermal | unavailable, explicitly recorded |
| Power | unavailable, explicitly recorded |

The dashboard therefore treats 4 threads as the current operator-profile
default. That is not a sustained-performance recommendation; it is the only
default with both a thread-envelope comparison and an operator-profile receipt
containing memory/storage context.

## Current Hot-Loop Boundary

The operator profile already records several reuse decisions:

```text
model_loaded_once = true
tokenizer_loaded_once = true
session_owned_buffers = true
prompt_token_buffer_reused = true
generated_token_buffer_reused = true
stop_policy_precomputed_once = true
stop_tail_buffer_reused = true
timing_buffers_reused = true
allocation_audit_buffers_reused = true
kv_cache_recreated_per_prompt = false
kv_cache_reused_across_prompts = true
kv_cache_cleared_per_prompt = true
prompt_token_cache_enabled = true
prompt_token_cache_policy = rendered_prompt_token_ids_reused_across_repeated_warm_session_prompts
sampler_recreated_per_prompt = false
sampler_reused_across_prompts = true
logits_buffer_reuse_claimed = true
```

The dashboard originally identified KV cache recreation, repeated tokenization,
and logits scratch allocation as hot-loop risks. Later receipts narrowed those
without changing the Qwen3 Q8_0 behavior oracle:

| Slice | Evidence | Result |
| --- | --- | --- |
| `SLM-CPU-031` | `qwen3-kv-cache-session-reuse.json` | One session KV cache is reused across prompts and cleared per prompt; prompt isolation remains explicit. |
| `SLM-CPU-032` | `qwen3-prompt-token-cache-validation.json` | Repeated rendered prompt IDs are cached; prompt-tokenize allocation bytes drop from `1063162466` to `531586554`; generated outputs and strict provenance match the baseline. |
| `SLM-CPU-035` | `qwen3-prefill-attribution-validation.json` | `prompt_prefill_breakdown.embed` and `prompt_prefill_breakdown.forward` are present; first-prompt `prompt_prefill.forward` allocation is `554666438` bytes and dominates `prompt_prefill.embed` at `157108` bytes. |

The next safe optimization slices should therefore start from these known
remaining costs:

1. Reduce `prompt_prefill.forward` / `model.forward` allocation and layout churn
   using the typed transformer-forward workspace boundaries already introduced
   after SLM-CPU-035.
2. Continue reducing `model.logits` tensor allocation and output-head costs.
   SLM-CPU-026 removes fresh full logits Vec allocation from default
   repetition-penalty decode steps by reusing a host scratch buffer, but the
   model still produces logits tensors per token.
3. Continue keeping sampler, stop policy, and tokenizer work out of the hot loop
   where doing so preserves deterministic generated IDs. SLM-CPU-029 reuses one
   sampler across prompts for the temperature-zero Qwen3 profile only, and
   SLM-CPU-032 caches repeated rendered prompt token IDs for the bounded warm
   session. Nonzero temperature modes still recreate samplers to avoid
   RNG-state coupling.
4. Improve Q8_0 dense linear locality only with before/after receipts proving
   identical prompt IDs, generated IDs, decoded text, backend identity,
   tokenizer authority, model SHA, and `fallback=false`.

SLM-CPU-045 starts the production-facing sidecar carrier for that fourth
boundary. The GGUF loader now carries inert Q8_0 sidecar descriptors alongside
the existing eager F32 Candle tensors, recording packed block counts, payload
byte counts, tensor roles, source/runtime shapes, offsets, and packed-byte
hashes. The descriptors are metadata-only:

```text
eager_f32_runtime_preserved = true
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
generated_id_preservation_required_before_runtime_use = true
```

The next runtime hook must be a behavior-preserving dense-linear dispatch
selector. It cannot select packed Q8_0 sidecar compute until generated IDs,
decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel,
model SHA, and `fallback=false` match the established Qwen3 Q8_0 appliance
oracle.

SLM-CPU-046 starts that selector boundary. The selector contract makes the
current runtime choice explicit:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
sidecar_candidate_status = missing | present_but_unavailable
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
equivalence_gate_required = true
```

Packed Q8_0 sidecar descriptors may be visible to the selector, but they remain
unavailable for runtime compute until a later slice proves generated-ID/text and
strict-receipt equivalence. This is an API/receipt boundary, not a performance
claim.

SLM-CPU-047 adds the next gate between the fixture-level Q8_0 sidecar prototype
and the production selector. The gate can record that a packed sidecar fixture
matches the eager F32 fixture output within a bounded tolerance, but it still
keeps production sidecar runtime compute disabled until generated-ID/text and
strict-receipt equivalence exist:

```text
artifact_kind = dense_gguf_q8_sidecar_equivalence_gate
fixture_equivalence_passed = true | false
generated_id_receipt_equivalence_passed = false
sidecar_runtime_compute_allowed = false
selected_kernel = dense-f32-candle-linear
speedup_claim = false
```

This narrows the remaining runtime blocker without changing generated IDs,
decoded text, tokenizer authority, backend/kernel identity, model SHA, or
`fallback=false`.

SLM-CPU-048 keeps the next step non-executing as well: it consumes the
equivalence gate and emits a runtime preflight report that names the remaining
blockers before packed Q8_0 sidecar compute can be selected:

```text
artifact_kind = dense_gguf_q8_sidecar_runtime_preflight
selected_path = eager_f32_candle
fixture_equivalence_passed = true | false
generated_id_receipt_equivalence_passed = false
production_compute_hook_available = false
sidecar_runtime_compute_allowed = false
runtime_blockers = generated_id_receipt_equivalence_missing, production_compute_hook_missing, production_selector_still_eager_f32
```

The preflight is the final eligibility surface before a later production
compute hook can be attached. It still does not execute packed sidecar compute
or claim speedup.

SLM-CPU-049 adds the generated-ID/text receipt equivalence gate required before
that preflight can stop naming generated-ID evidence as a blocker. The gate
compares an eager F32 behavior oracle receipt against a future packed Q8_0
sidecar candidate receipt across the fields that matter for this lane:

```text
artifact_kind = dense_gguf_q8_generated_id_text_equivalence
model_sha256 = equal
tokenizer_source = equal
tokenizer_strict = true
corpus_id / prompt_id = equal
prompt_ids = equal
generated_ids = equal
decoded_text = equal
selected_backend = equal
selected_kernel = equal
fallback_used = false
speedup_claim = false
sidecar_runtime_compute_allowed = false
```

Passing this gate only proves behavior/provenance equivalence for the compared
receipts. It still keeps packed sidecar runtime compute disabled until a later
production compute hook exists and the selector is explicitly updated under the
same receipt discipline.

SLM-CPU-050 adds the production-compute-hook availability surface. This is
still a non-executing report: it distinguishes a missing production compute hook
from a hook that exists but is not yet selected by the runtime selector. The
selected path remains the eager F32 Candle oracle until a later selector update
is proven with generated-ID/text receipts:

```text
artifact_kind = dense_gguf_q8_production_compute_hook
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
hook_status = missing | available_but_selector_still_eager_f32
generated_id_receipt_equivalence_passed = true | false
production_compute_hook_available = true | false
selector_update_required_before_runtime_use = true
sidecar_runtime_compute_allowed = false
eager_f32_runtime_preserved = true
dense_runtime_replaced = false
speedup_claim = false
```

The availability surface does not execute packed Q8_0 sidecar compute, replace
the dense runtime, change generated IDs/text, or claim speedup. It only makes
the next runtime blocker machine-checkable.

SLM-CPU-051 adds the selector-readiness gate after the hook availability
surface. It is still non-executing: the report can state that generated-ID/text
equivalence and a production compute hook are both present, but it does not
change the runtime selector itself. A later selector-update PR must carry the
actual runtime behavior change and its before/after receipts.

```text
artifact_kind = dense_gguf_q8_selector_readiness_gate
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
readiness_status = blocked | ready_for_separate_selector_update
selector_update_ready = true | false
selector_update_required_before_runtime_use = true
sidecar_runtime_compute_allowed = false
runtime_blockers = production_selector_still_eager_f32 | ...
eager_f32_runtime_preserved = true
dense_runtime_replaced = false
speedup_claim = false
```

Readiness here means only that the next selector PR has the required proof
inputs. It is not packed Q8_0 production execution and it is not a speedup
claim.

SLM-CPU-052 is that next selector-update gate. It starts from the SLM-CPU-051
readiness artifact and adds an explicit selector-update receipt plus an opt-in
selector path for the packed Q8_0 sidecar candidate. The ordinary dense Q8_0
selector still preserves eager F32 Candle unless the selector-update proof is
supplied.

```text
artifact_kind = dense_gguf_q8_selector_update
previous_selected_path = eager_f32_candle
selected_path = packed_q8_sidecar | eager_f32_candle
selected_kernel = dense-q8-sidecar-linear | dense-f32-candle-linear
selector_update_applied = true | false
sidecar_runtime_compute_allowed = true | false
runtime_blockers = [] | ...
speedup_claim = false
```

The update may select the packed sidecar only where the eager F32 Candle oracle
and the packed Q8_0 candidate preserve the same prompt IDs, generated IDs,
decoded text, strict tokenizer authority, selected CPU backend/kernel, model
SHA, and `fallback=false`. It is a behavior-preserving selector update, not a
sustained-throughput or portable CPU-performance claim.

SLM-CPU-053 is the next runtime gate. It may enable packed Q8_0 sidecar compute
only for the exact evidence-scoped tensor or minimal tensor set covered by the
selector-update proof, and only with before/after receipts proving unchanged
model SHA, tokenizer source and strictness, prompt IDs, generated IDs, decoded
text, selected CPU backend/kernel identity, and `fallback=false`. If that proof
cannot be produced, the runtime must stay on eager F32 Candle and record the
specific blocker. This remains a behavior-preserving Kaby appliance slice, not
a sustained-throughput, Q4/Q5, accelerator, Qwen3.5, server, or BitNet QK256
claim.

The first SLM-CPU-053 runtime gate records the blocker instead of enabling
packed sidecar compute:

```text
artifact_kind = dense_gguf_q8_runtime_execution_proof
execution_status = blocked
selector_update_status = applied_to_packed_sidecar_candidate
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
production_runtime_hook_invoked = false
runtime_compute_enabled = false
sidecar_runtime_compute_allowed = false
runtime_blockers =
  production_dispatch_still_eager_f32
  packed_runtime_compute_disabled
  production_runtime_hook_missing
  before_after_receipts_missing
eager_f32_runtime_preserved = true
dense_runtime_replaced = false
fallback_used = false
speedup_claim = false
```

This closes only the runtime proof gate for the current state: packed Q8_0
sidecar compute is still disabled until a later PR adds the production forward
hook and after-execution receipts for the same Qwen3 Q8_0 behavior oracle.

SLM-CPU-054 makes that remaining hook/API gap machine-checkable instead of
treating it as an implicit prose blocker. It consumes the SLM-CPU-053 runtime
proof shape and records the exact production gaps that still prevent packed
Q8_0 sidecar compute from being selected:

```text
artifact_kind = dense_gguf_q8_runtime_hook_gap
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
production_runtime_hook_invoked = false
runtime_compute_enabled = false
sidecar_runtime_compute_allowed = false
api_gaps =
  production_dispatch_still_eager_f32
  packed_runtime_compute_still_disabled
  transformer_dense_linear_hook_missing
  before_after_receipt_capture_missing
fallback_used = false
speedup_claim = false
```

The next safe step remains a production dense-linear hook that receives the
selected Q8_0 sidecar descriptor and emits before/after Qwen3 Q8_0 behavior
receipts before enabling packed sidecar compute. SLM-CPU-054 does not replace
the eager F32 Candle runtime path and does not claim speedup.

SLM-CPU-055 is the next contract gate for that step. It should add the first
production dense-linear hook boundary so transformer dense linear calls can
receive either an explicit eager-F32 selection or a selected Q8_0 sidecar
descriptor. The default runtime must remain eager F32 Candle unless before/after
receipts prove identical model SHA, tokenizer authority, prompt IDs, generated
IDs, decoded text, selected CPU backend/kernel identity, and `fallback=false`.
If packed compute remains disabled, the slice must emit a machine-checkable
hook-contract or blocker artifact instead of making a speedup claim.

SLM-CPU-056 implements that hook boundary in the production transformer path.
Dense Q8_0 sidecar descriptors can now reach transformer attention and MLP
dense-linear calls as an explicit hook registry, and each call records whether
the selected descriptor is present. The runtime selection still remains:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
```

The machine-checkable artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-19/qwen3-production-dense-linear-hook-boundary.json`
records the remaining packed-compute and receipt gaps. The slice does not
enable packed Q8_0 sidecar execution or claim any performance improvement.

SLM-CPU-057 adds the next receipt gate on top of that boundary. Qwen3 Q8_0
warm-session aggregate and per-prompt receipts now carry a
`dense_q8_hook_selection` object so a future packed sidecar candidate must
preserve the hook-selection identity as well as model SHA, tokenizer
source/strictness, prompt IDs, generated IDs, decoded text, selected CPU
backend/kernel identity, and `fallback=false`.

The default runtime still remains:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
runtime_compute_enabled = false
dense_runtime_replaced = false
speedup_claim = false
```

The machine-checkable blocker artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-19/qwen3-dense-hook-receipt-gate.json`
records that packed Q8_0 sidecar compute is still disabled until a packed
compute kernel exists and before/after Qwen3 Q8_0 warm-session receipts prove
identical generated output and dense hook-selection identity.

SLM-CPU-058 queued the before/after receipt gate. It records that packed Q8_0
sidecar compute remains disabled until a selected dense-hook path can be
compared against the eager F32 Candle behavior oracle by receipts that match on
model SHA, tokenizer source and strictness, prompt IDs, generated IDs, decoded
text, selected CPU backend/kernel identity, dense hook-selection identity, and
`fallback=false`. A speedup or sustained-throughput claim remains out of scope.

SLM-CPU-059 is the next compute-kernel proof gate. The committed blocker
artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-20/qwen3-packed-q8-compute-kernel-proof-gate.json`
names the current production gap: transformer dense-linear hooks still receive
metadata-only sidecar descriptors, not payload-bearing packed Q8_0 blocks, so
no production dense-linear call can safely execute a packed dequant-fused
matvec yet. The next safe runtime slice is to extend the hook contract with an
evidence-scoped payload-bearing sidecar candidate and prove one tensor path with
before/after Qwen3 Q8_0 warm-session receipts before broader selection or
timing claims.

SLM-CPU-060 adds that payload-bearing hook contract without enabling runtime
compute. `DenseLinearRuntimeHookDescriptor` can now carry an optional
`DenseLinearPackedQ8Payload` for a single evidence-scoped tensor path, and the
transformer boundary reports payload byte count, Q8 block count, matrix shape,
and contract validity. The selected production path remains `eager_f32_candle`
with `dense-f32-candle-linear`, `runtime_compute_enabled=false`,
`dense_runtime_replaced=false`, and `speedup_claim=false`. The committed
contract artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-20/qwen3-payload-bearing-q8-sidecar-hook-contract.json`
records the remaining gate: production loading still attaches metadata-only
descriptors by default, and one exact Qwen3 Q8_0 tensor path must be wired
behind an opt-in gate and proved by before/after warm-session receipts before
packed Q8_0 sidecar compute can be selected.

SLM-CPU-061 wires one exact real Qwen3 Q8_0 dense-linear tensor into that
payload-bearing hook contract behind an explicit opt-in gate. With
`BITNET_DENSE_Q8_PAYLOAD_ENABLE=1` and
`BITNET_DENSE_Q8_PAYLOAD_TENSOR=blk.0.attn_q.weight`, the production GGUF load
attaches the packed bytes for `layers.0.attention.q_proj.weight` to a single
`AttentionQ` hook boundary. The committed before/after warm-session receipts
show the selected path remains `eager_f32_candle`, selected kernel remains
`dense-f32-candle-linear`, `runtime_compute_enabled=false`,
`dense_runtime_replaced=false`, and generated outputs are unchanged. The
summary artifact
`ci/slm-cpu/intel-i5-8250u/2026-05-20/qwen3-real-q8-sidecar-payload-candidate.json`
records matching model SHA, strict GGUF tokenizer authority, prompt IDs,
generated IDs, decoded text, selected CPU backend/kernel identity, and
`fallback=false`. It does not claim packed Q8 compute, speedup, sustained
throughput, Q4/Q5 support, accelerator execution, Qwen3.5 support, or BitNet
QK256 changes.

## Greedy Sampler Fast Path

SLM-CPU-024 adds a guarded sampler fast path for `temperature = 0.0` when
repetition penalty is inactive, or when there is no context to penalize. The
sampler returns the greedy argmax directly instead of copying logits into its
scratch buffer first.

The after-change validation compares the new 4-thread warm-session receipt
against the SLM-CPU-015 4-thread baseline:

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-15/qwen3-warm-session-threads-4.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path-validation.json
generated_outputs_match_baseline = true
sampler_decode_allocations_zero = true
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This closes only the greedy no-penalty sampler scratch-copy boundary. It does
not remove the remaining `model.logits_and_extract` allocation, change Q8_0
dense math, or establish a sustained throughput claim.

## Logits Extraction Isolation

SLM-CPU-025 narrows the next allocation boundary without changing generated
tokens. In the guarded deterministic greedy/no-penalty case it selects the
argmax directly from the logits tensor, bypassing full host `Vec<f32>` logits
extraction. For default warm-session steps with active repetition penalty, it
keeps the existing vector extraction path because that path is still required
for count-aware repetition-penalty semantics.

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-greedy-sampler-fast-path.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse-validation.json
generated_outputs_match_baseline = true
direct_greedy_logits_steps = 6
logits_vec_extraction_steps = 40
logits_vec_extraction_bypassed_for_all_steps = false
model_logits_and_extract_alloc_bytes_delta = -3643896
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This is an isolation slice, not a full logits-buffer reuse claim. The remaining
vector extraction steps are explicit in the receipt and should only be removed
after the repetition-penalty path has an allocation-safe equivalent that
preserves generated IDs.

## Repetition-Penalty Logits Reuse

SLM-CPU-026 adds that allocation-safe repetition-penalty equivalent for the
warm-session path. Count-aware repetition penalties are still applied before
greedy selection, but default decode steps now copy CPU F32 logits into a
caller-owned scratch buffer and sample in place instead of materializing a fresh
host `Vec<f32>` per token.

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-logits-extraction-reuse.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse-validation.json
generated_outputs_match_baseline = true
direct_greedy_logits_steps = 6
logits_scratch_reuse_steps = 40
logits_vec_extraction_steps = 0
logits_vec_extraction_bypassed_for_all_steps = true
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This still does not claim sustained throughput or dense math acceleration. It
only narrows the host allocation boundary for the existing Qwen3 Q8_0 CPU
behavior oracle.

## Warm-Session Sampler Reuse

SLM-CPU-029 removes the remaining per-prompt sampler object recreation in the
bounded Qwen3 Q8_0 warm-session appliance profile. The reuse is deliberately
guarded to `temperature = 0.0`, where the sampler does not need cross-prompt RNG
state. Sampling modes with nonzero temperature retain per-prompt sampler
creation for request independence.

```text
baseline = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-repetition-penalty-logits-reuse.json
after = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-kv-temp-reuse.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-17/qwen3-kv-temp-reuse-validation.json
generated_outputs_match_baseline = true
sampler_recreated_per_prompt = false
sampler_reused_across_prompts = true
sampler_reused_prompt_count = 6
sampler_recreated_prompt_count = 0
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This slice does not change KV-cache isolation: the KV cache is still recreated
per prompt to preserve prompt independence. It also does not claim a measured
speedup; allocation counters for the dominant tensor-producing components are
unchanged, and the evidence is scoped to removal of avoidable sampler setup in
the existing 4-thread appliance profile.

## Warm-Session KV Cache Reuse

SLM-CPU-031 reuses one CPU KV cache across Qwen3 Q8_0 warm-session prompts and
clears it before each prompt. The reuse is scoped to the resident session and
keeps prompt isolation explicit; it moves the large KV-cache tensor allocation
out of per-prompt `prompt_setup` and records it once as session setup.

```text
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-kv-cache-session-reuse.json
generated_outputs_match_baseline = true
quality_passed = true
determinism_passed = true
kv_cache_recreated_per_prompt = false
kv_cache_reused_across_prompts = true
kv_cache_cleared_per_prompt = true
kv_cache_reused_prompt_count = 6
session_setup_kv_cache_alloc_bytes = 9395257760
prompt_setup.kv_cache_alloc_bytes_per_first_prompt = 0
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

This does not claim sustained throughput or a portable performance result. It
only narrows the resident-session allocation boundary for the existing Qwen3
Q8_0 4-thread Kaby appliance profile.

## Prompt Token Cache

SLM-CPU-032 reuses rendered prompt token IDs inside the resident warm session.
The reuse is scoped to repeated prompts with the same rendered prompt, BOS
policy, and special-token parse policy. It keeps the generated-ID oracle intact
while avoiding redundant `tokenizer.encode` work for repeated corpus cases.

```text
prompt_token_cache_policy = rendered_prompt_token_ids_reused_across_repeated_warm_session_prompts
prompt_token_cache_enabled = true
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json
prompt_token_cache_hits = 3
prompt_token_cache_misses = 3
prompt_token_cache_entry_count = 3
generated_outputs_match_baseline = true
quality_passed = true
determinism_passed = true
fallback_used = false
before_prompt_tokenize_alloc_bytes = 1063162466
after_prompt_tokenize_alloc_bytes = 531586554
speedup_claim = false
sustained_throughput_claim = false
```

This is a tokenizer-boundary optimization only. It does not change dense math,
does not claim Q4/Q5 support, and does not turn the bounded Kaby appliance
profile into a sustained-throughput result.

## Aggregate Allocation Boundary

SLM-CPU-033 records the next warm-session allocation target directly in the
aggregate receipt. After prompt-token caching, the real i5-8250U evidence still
shows the largest allocation-counter deltas in prompt prefill and dense model
forward execution:

```text
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prompt-token-cache-validation.json
dominant_hotspot = prompt_prefill
next_optimization_target = prefill_model_forward_allocation_attribution
fallback_used = false
speedup_claim = false
sustained_throughput_claim = false
```

The target is diagnostic. It tells the next runtime slice to attribute
`prompt_prefill` and `model.forward` tensor allocation before changing kernels
or dense math. It does not prove a speedup and it does not broaden the Kaby
claim boundary.

## Prompt Prefill Attribution

SLM-CPU-034 keeps the existing `prompt_prefill` allocation total for receipt
continuity and adds nested subcomponent counters:

```text
prompt_prefill_breakdown.embed
prompt_prefill_breakdown.forward
ranked_hotspots includes prompt_prefill.embed and prompt_prefill.forward
```

This is attribution only. It identifies whether the prefill hotspot is coming
from token embedding or dense model forward execution before any Q8_0 GEMV,
RMSNorm, RoPE, or attention-loop optimization is attempted.

## Prefill Attribution Artifact

SLM-CPU-035 records a real i5-8250U Qwen3 Q8_0 warm-session receipt after the
prompt-prefill attribution slice:

```text
evidence = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prefill-attribution.json
validation = ci/slm-cpu/intel-i5-8250u/2026-05-18/qwen3-prefill-attribution-validation.json
generated_outputs_match_baseline = true
decoded_text_matches_baseline = true
fallback_used = false
tokenizer_source = gguf_metadata
tokenizer_strict = true
first_prompt_prefill_embed_alloc_bytes = 157108
first_prompt_prefill_forward_alloc_bytes = 554666438
```

The receipt preserves the behavior oracle from the prompt-token cache baseline
while adding `prompt_prefill_breakdown.embed`,
`prompt_prefill_breakdown.forward`, and ranked prompt-prefill subcomponent
hotspots. It is evidence for attribution only: it does not claim a runtime
speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

## Prefill Forward Buffer Boundary

SLM-CPU-036 classifies the first reusable allocation surface under
`prompt_prefill.forward` before changing dense math. The warm-session receipt
first recorded the boundary as transformer forward workspace and owned tensor
outputs that were not caller-reusable from the CLI warm-session loop:

```text
next_optimization_target = prefill_forward_buffer_boundary
first_reusable_allocation_surface = transformer_forward_workspace_api_and_owned_tensor_outputs
claim_scope = allocation-boundary classification only
```

SLM-CPU-037 keeps that boundary explicit instead of adding a broad caller-side
buffer reuse patch. SLM-CPU-038 introduces the first typed transformer forward
workspace API boundary through `TransformerModel::forward_with_workspace`,
`TransformerBlock::forward_with_workspace`, and
`FeedForward::forward_with_workspace`. This first API slice still delegates
tensor math to the existing owned-output path, so reusable tensor storage is not
enabled yet. The receipt therefore records:

```text
required_api_boundary = typed_transformer_forward_workspace
optimization_deferred = true
```

SLM-CPU-039 narrows that boundary to the first workspace-owned transformer
output surface. `FeedForward::forward_with_workspace` now routes the owned
feed-forward output tensor through `TransformerForwardWorkspace` before
returning it to the existing block math. Candle still constructs the tensor
through the existing owned-output path, so this is ownership plumbing and
allocation attribution, not reusable storage or a speedup claim:

```text
first_reusable_allocation_surface = feed_forward.output
workspace_storage_owner = TransformerForwardWorkspace
reuse_status = feed_forward_output_workspace_owned_reuse_not_enabled
next_optimization_target = feed_forward_output_workspace_owned_boundary
status = workspace_owned_output_reuse_deferred
optimization_deferred = true
```

SLM-CPU-040 reaches the next narrower hook: the exact
`FeedForward::down_proj` output boundary. The workspace observes this boundary
before the output is returned, but reusable storage remains blocked because
`candle_nn::Linear::forward` constructs and returns a new `Tensor`; it does not
expose an out-parameter or reusable output-storage API that can be filled by
`TransformerForwardWorkspace` without changing the linear implementation.

```text
first_reusable_allocation_surface = feed_forward.down_proj.output
workspace_storage_owner = TransformerForwardWorkspace
reuse_status = feed_forward_down_proj_output_storage_reuse_blocked_by_candle_linear
next_optimization_target = feed_forward_down_proj_output_storage_boundary
status = down_proj_output_storage_reuse_blocked
optimization_deferred = true
```

The next safe implementation target is adding or adopting a
behavior-preserving linear output-storage API before replacing
`FeedForward::down_proj` output construction with reusable workspace-backed
storage. This remains an allocation-boundary proof only; it makes no speedup,
sustained-throughput, broad-answer-quality, Q4/Q5 runtime, accelerator,
Qwen3.5, or BitNet QK256 claim.

SLM-CPU-041 narrows the blocker from "Linear is opaque" to the exact Candle
tensor API gap. `candle_nn::Linear` exposes `weight()` and `bias()`, so the
Qwen3 `FeedForward::down_proj` shapes can be inspected without changing math.
The behavior-preserving compute path still uses `Tensor::matmul` plus optional
`broadcast_add`, and those operations return owned tensors without a
caller-provided output-storage parameter. Reusable workspace-backed storage is
therefore still deferred until Candle grows, or this repo adopts, an explicit
matmul/bias-add output-storage API:

```text
first_reusable_allocation_surface = feed_forward.down_proj.output
workspace_storage_owner = TransformerForwardWorkspace
reuse_status = dense_linear_output_storage_blocked_by_candle_tensor_ops
required_api_boundary = dense_linear_output_storage_api_boundary
weight_accessible = true
bias_accessible = true
can_fill_caller_output_storage = false
optimization_deferred = true
```

This is a classification slice. It does not change Q8_0 GEMV, RMSNorm, RoPE,
attention, output-head math, tokenizer behavior, or generated tokens, and it
does not claim a speedup, sustained throughput, Q4/Q5 runtime support,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

SLM-CPU-042 moves the next target from reusable output storage to the first
Q8_0 dense linear locality boundary that can be inspected without changing
runtime behavior. The current dense standard GGUF load path eagerly dequantizes
Q8_0 blocks into a host `Vec<f32>`, reshapes projection and embedding tensors
for Candle without transposing the dequantized values, and then constructs a
Candle tensor from that F32 slice:

```text
locality_boundary = eager_dense_standard_quant_dequant_to_f32_before_candle_tensor
dequantizes_before_compute = true
materializes_f32_tensor = true
values_transposed = false
shape_reshaped_without_transpose = true for GGML projection/embedding layouts
next_optimization_target = q8_dense_linear_locality_boundary
```

This boundary is instrumentation only. A later runtime optimization may replace
the eager F32 materialization or improve Q8_0 dense linear locality only if it
preserves generated IDs, decoded text, strict GGUF tokenizer authority, selected
CPU backend/kernel, model SHA, and `fallback=false` in before/after receipts.
It does not reopen the Candle output-storage blocker, claim a speedup, or
broaden support to Q4/Q5, accelerators, Qwen3.5, or BitNet QK256.

SLM-CPU-043 starts that later slice with a fixture-level Q8_0 sidecar
prototype. The prototype keeps packed Q8_0 block scales and signed codes,
computes a narrow CPU matvec by dequantizing inside the dot product, and
compares the result against the existing eager F32 fixture output:

```text
artifact_kind = dense_gguf_q8_linear_sidecar_prototype
dequantizes_inside_matvec = true
materializes_full_f32_weights = false
compares_against_eager_f32_reference = true
dense_runtime_replaced = false
speedup_claim = false
```

This is not yet a production transformer path. It proves the first local
packed-Q8 sidecar compute boundary can match the eager F32 fixture for a single
dense linear role. A runtime replacement still requires before/after warm-session
receipts proving unchanged generated IDs, decoded text, strict tokenizer
authority, selected CPU backend/kernel, model SHA, and `fallback=false`.

SLM-CPU-067 promotes that boundary to one exact real Qwen3 tensor path:
`layers.0.attention.q_proj.weight`, sourced from `blk.0.attn_q.weight`. The
runtime hook remains opt-in and exact-tensor gated. By default, the transformer
continues to use the eager F32 Candle linear path. Packed sidecar execution is
selected only when all four environment gates name the same evidence-scoped
tensor:

```text
BITNET_DENSE_Q8_PAYLOAD_ENABLE=1
BITNET_DENSE_Q8_PAYLOAD_TENSOR=blk.0.attn_q.weight
BITNET_DENSE_Q8_RUNTIME_ENABLE=1
BITNET_DENSE_Q8_RUNTIME_TENSOR=blk.0.attn_q.weight
```

The SLM-CPU-067 Kaby artifact bundle records before/after strict warm-session
receipts for the verified Qwen3 Q8_0 appliance profile:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-067-before-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-067-after-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-067-runtime-hook-equivalence.json
```

The baseline receipt reports `selected_path = eager_f32_candle`. The opt-in
receipt reports `selected_path = packed_q8_sidecar`,
`selected_kernel = dense-q8-sidecar-linear`, and
`runtime_compute_enabled = true` for only the exact Q projection tensor. The
equivalence artifact records matching prompt IDs, generated IDs, decoded text,
model SHA, strict GGUF tokenizer authority, selected CPU backend, and
`fallback=false` across both receipts.

This is a behavior-preserving exact-tensor runtime hook candidate. It does not
enable packed Q8_0 runtime compute by default, does not generalize packed Q8_0
execution to all dense tensors, and does not claim speedup, sustained
throughput, broad answer quality, Q4/Q5 runtime support, accelerator execution,
Qwen3.5 support, or BitNet QK256 changes.

SLM-CPU-068 classifies the timing evidence from that exact same artifact pack.
The result is intentionally conservative: the hook preserved generated IDs and
decoded text, but the opt-in packed sidecar path regressed on the bounded
two-prompt 4-thread receipt.

```text
classification = regressed_on_bounded_two_prompt_artifact
before_selected_path = eager_f32_candle
after_selected_path = packed_q8_sidecar
behavior_equivalence.passed = true
runtime_promotion_recommended = false
speedup_claim = false

before.total_session_ms = 118266.381
after.total_session_ms = 138900.052
delta.total_session_ms = +20633.671 (+17.446%)

before.warm_prompt_wall_ms = 41085.124
after.warm_prompt_wall_ms = 57917.182
delta.warm_prompt_wall_ms = +16832.058 (+40.966%)

before.decode_generated_tok_s = 1.225
after.decode_generated_tok_s = 0.956
delta.decode_generated_tok_s = -0.269 (-21.959%)
```

The timing classification artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-068-exact-hook-timing-classification.json`.
The default runtime should remain `eager_f32_candle`. The packed sidecar hook is
still useful as a correctness-preserving implementation boundary for further
locality and kernel work, but it is not a promotion candidate in its current
form.

SLM-CPU-069 localizes the next packed-Q8 sidecar work from that regression
without changing runtime defaults. The machine-checkable root-cause artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-069-packed-q8-locality-root-cause.json`.
It records that the regression is concentrated in prefill/forward timing while
logits and sampling are stable, and classifies the likely root cause as
`packed_block_decode_and_matvec_locality`.

The current opt-in sidecar kernel is a scalar reference matvec over the packed
Q8_0 bytes for only `layers.0.attention.q_proj.weight`. It decodes the fp16
Q8_0 block scale inside the innermost per-weight loop and does not reuse the
scale across the 32 codes in each block. The next safe target is therefore a
block-local matvec prototype that decodes each block scale once per block and
keeps behavior gated against the eager F32 oracle before any timing
interpretation.

```text
root_cause.primary = packed_block_decode_and_matvec_locality
root_cause.secondary = scratch_allocation
root_cause.unlikely = selector_overhead, receipt_timing_instrumentation
next_target = SLM-CPU-070 packed Q8 block-local matvec prototype
default_runtime_changed = false
speedup_claim = false
```

SLM-CPU-070 adds that prototype in the exact-tensor opt-in sidecar helper
without promoting it to the default runtime. The implementation now walks
contiguous Q8_0 block spans and decodes each fp16 scale once per block segment
instead of once per individual weight. The code-level artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-070-packed-q8-block-local-matvec-prototype.json`.

The prototype remains a behavior-gated implementation slice, not a performance
claim. The verified Qwen3 GGUF was not present in the development workspace for
this slice, so the committed evidence is limited to the exact dense-linear
reference tests. Real i5-8250U before/after receipt equivalence is still
required before any timing interpretation or promotion decision.

```text
selected_default_path = eager_f32_candle
opt_in_candidate_path = packed_q8_sidecar
exact_tensor = layers.0.attention.q_proj.weight
scale_decode = once per contiguous Q8 block segment
code_level_reference_tests = passed
row_split_q8_block_reference_test = passed
real_qwen3_generated_id_receipts_regenerated = false
speedup_claim = false
```

SLM-CPU-071 is the required real-artifact gate after that prototype. It must
regenerate the verified Qwen3-0.6B Q8_0 i5-8250U 4-thread warm-session
before/after receipts with the default eager F32 path and the opt-in exact
`layers.0.attention.q_proj.weight` packed sidecar candidate. The first gate is
behavioral: prompt IDs, generated IDs, decoded text, strict GGUF tokenizer
authority, selected CPU backend/kernel, model SHA, hook identity, and
`fallback=false` must match before timing is classified as improved, regressed,
or inconclusive. This gate still does not enable packed Q8_0 by default or claim
sustained throughput.

SLM-CPU-071 merged the gate definition, not the actual before/after artifact
pack. No `qwen3-slm-cpu-071-*` warm-session receipts are committed under
`ci/slm-cpu/intel-i5-8250u/2026-05-21/`. SLM-CPU-072 is therefore the real
artifact-capture follow-up: it must either commit the before/after receipts and
classification for the exact Qwen3 Q8_0 appliance profile, or record that the
verified GGUF was unavailable in the execution environment. It must not treat
the merged gate definition as timing evidence.

## Claim Boundary

This dashboard may be used to claim:

```text
Qwen3-0.6B Q8_0 has a bounded i5-8250U strict CPU performance baseline.
The current operator profile uses 4 threads.
Generated IDs were stable across the 1/2/4/8 thread envelope.
Memory and storage context are present for the 4-thread operator profile.
Thermal and power fields are present but unavailable.
```

This dashboard must not be used to claim:

```text
sustained 8250U throughput
broad chat quality
Q4/Q5 support
server inference
GPU, NPU, OpenVINO, or UHD 620 acceleration
Qwen3.5 support
BitNet QK256 changes
portable performance across other CPUs
```

## Release-Surface Boundary

After SLM-CPU-061, BitNet-rs keeps the Kaby Lake SLM lane as the audited
release and evidence surface. The committed Qwen3 Q8_0 and Qwen2.5 Q8_0
strict CPU receipts remain the behavior oracle for any future dense packed-Q8
candidate.

Further packed Q8_0 compute-candidate development should happen in
`bitnet-rs-swarm`. A candidate can return to BitNet-rs only as an audited
release/evidence artifact that preserves:

```text
model SHA
strict GGUF tokenizer authority
prompt IDs
generated IDs
decoded text
selected CPU backend/kernel identity
dense hook-selection identity
fallback_used=false
speedup_claim=false unless a separate bounded timing receipt proves otherwise
```

This boundary does not claim a new runtime compute path, speedup, sustained
throughput, broad answer quality, Q4/Q5 runtime support, server execution,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

The intake rules for those returned artifacts are defined in
`docs/slm/SLM_CPU_SWARM_ARTIFACT_INTAKE.md`.

## SLM-CPU-065 Runtime Promotion Gate

SLM-CPU-065 reviewed the accepted single-tensor packed Q8_0 sidecar candidate
from the SLM-CPU-064 intake package:

```text
tensor = layers.0.attention.q_proj.weight
role = AttentionQ
payload_sha256 = a8e16a232ea8c19a5c5d2eb5f21bfdf5c297eba0ac90e74b0afc052577179c24
```

The release-surface decision is blocked, not promoted. BitNet-rs keeps the
runtime on the strict eager F32 Candle path until an exact-tensor packed-Q8
runtime receipt exists:

```text
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
runtime_compute_enabled = false
dense_runtime_replaced = false
fallback_used = false
speedup_claim = false
```

The gate artifact is
`ci/slm-cpu/intel-i5-8250u/2026-05-20/slm-cpu-065-runtime-promotion-gate.json`.
It records that the sidecar evidence was accepted, but runtime promotion is
blocked because the release surface has not produced a strict before/after
receipt where packed sidecar compute is selected for this exact tensor and
behavior remains identical.

This gate does not claim packed Q8_0 execution, speedup, sustained throughput,
broad answer quality, Q4/Q5 runtime support, server execution, accelerator
execution, Qwen3.5 support, or BitNet QK256 changes.

## SLM-CPU-074 Packed Q8 Sidecar Instrumentation

SLM-CPU-074 adds aggregate runtime counters for the opt-in exact-tensor
`layers.0.attention.q_proj.weight` packed Q8_0 sidecar path. The counters are
available from `bitnet-transformer` through
`dense_q8_sidecar_instrumentation_snapshot()` and can be reset with
`reset_dense_q8_sidecar_instrumentation()`.

The instrumentation measures the costs that SLM-CPU-073 identified as the next
blocking surface:

```text
selector dispatch calls and elapsed ns
selector selected / declined / error counts
input materialization calls, elapsed ns, and value count
bias extraction calls, elapsed ns, and value count
packed matvec calls, elapsed ns, input rows, and output values
output tensor construction calls and elapsed ns
```

The default production path remains `eager_f32_candle`. Packed Q8_0 sidecar
execution remains opt-in, payload-gated, and exact-tensor scoped. These counters
are diagnostic evidence only; they do not claim speedup, sustained 8250U
throughput, broad answer quality, Q4/Q5 runtime support, server execution,
accelerator execution, Qwen3.5 support, or BitNet QK256 changes.

## SLM-CPU-075 Instrumentation Artifact Boundary

SLM-CPU-075 records the first post-instrumentation diagnostic artifact:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-21/qwen3-slm-cpu-075-packed-q8-instrumentation-artifact.json
```

The artifact consumes the SLM-CPU-074 counter surface and keeps the prior
SLM-CPU-072 before/after receipts as the behavior oracle. It classifies selector
dispatch, input materialization, bias extraction, packed matvec compute, and
output tensor construction as instrumented surfaces, but it does not claim real
end-to-end counter values because the warm-session receipt path does not yet
snapshot and serialize the transformer-side aggregate counters around a bounded
Qwen3 Q8_0 run.

The resulting blocker is narrow: add a warm-session sidecar instrumentation
receipt bridge before using these counters to drive another packed-Q8 runtime
promotion or optimization. That bridge must reset the counters before the
opt-in exact-tensor run, snapshot them afterward, and prove generated IDs,
decoded text, model SHA, tokenizer source and strictness, selected CPU backend
identity, dense hook identity, and `fallback_used=false` remain unchanged.

The default path remains `eager_f32_candle`; packed Q8_0 sidecar execution
remains opt-in and exact-tensor scoped to `layers.0.attention.q_proj.weight`.
SLM-CPU-075 makes no speedup, sustained-throughput, broad answer-quality,
Q4/Q5 runtime-support, server, accelerator, Qwen3.5, or BitNet QK256 claim.

## SLM-CPU-076 Warm-Session Instrumentation Bridge

SLM-CPU-076 bridges the packed Q8_0 sidecar instrumentation counters into the
Qwen3 Q8_0 warm-session aggregate receipt. The warm-session command resets the
`bitnet-transformer` counters before the prompt loop and snapshots them after
the bounded run, then records the result under
`dense_q8_sidecar_instrumentation`.

The receipt bridge serializes:

```text
selector dispatch / selected / declined / error counters
input materialization calls, elapsed ns, and value count
bias materialization calls, elapsed ns, and value count
packed matvec calls, elapsed ns, input rows, and output values
output tensor construction calls and elapsed ns
```

It also records that the sidecar path remains opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight`, while the default runtime remains
`eager_f32_candle`. The bridge is diagnostic only. It does not enable packed
Q8_0 sidecar execution by default, broaden the hook beyond the exact tensor,
or claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime
support, server execution, accelerator execution, Qwen3.5 support, or BitNet
QK256 changes.

## SLM-CPU-077 Post-Bridge Counter Artifact

SLM-CPU-077 consumes that bridge with a real i5-8250U Qwen3 Q8_0 warm-session
sidecar run. The committed artifact pack is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-077-post-bridge-packed-q8-sidecar-warm-session.json
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-077-post-bridge-counter-classification.json
```

The sidecar run remains opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight`. The companion classification compares the
new sidecar receipt against the committed SLM-CPU-072 eager F32 oracle because
a same-turn eager oracle rerun failed on the low-free-space Kaby host before
receipt write with:

```text
memory allocation of 167772160 bytes failed
```

The behavior oracle still passes for the compared receipts:

```text
prompt_ids_match = true
generated_ids_match = true
decoded_text_match = true
model_sha_match = true
tokenizer_source_match = true
tokenizer_strict_match = true
selected_backend_match = true
fallback_false_before_after = true
```

The serialized counter pack names the dominant measured sidecar cost:

```text
selector_dispatch_calls = 42336
selector_selected_calls = 216
selector_declined_calls = 42120
selector_error_calls = 0
input_materialization_calls = 216
bias_materialization_calls = 216
packed_matvec_calls = 216
output_tensor_construction_calls = 216
packed_matvec_ns = 19093586100
```

The classification is therefore:

```text
next_target = packed_matvec_compute
runtime_promotion_recommended = false
default_runtime_changed = false
speedup_claim = false
```

SLM-CPU-077 does not enable packed Q8_0 by default, broaden execution beyond
the exact tensor, claim speedup, claim sustained throughput, start Q4/Q5
runtime support, or involve server, accelerator, Qwen3.5, or BitNet QK256 work.

## SLM-CPU-078 Packed Matvec Compute Target

SLM-CPU-078 is the next queued implementation target after the SLM-CPU-077
counter artifact. It is scoped to reducing or further classifying
`packed_matvec_compute` for the opt-in exact-tensor packed Q8_0 sidecar path.

Any SLM-CPU-078 runtime change must preserve the Qwen3 Q8_0 appliance oracle
before making even a bounded improvement claim:

```text
model SHA unchanged
strict GGUF tokenizer authority unchanged
prompt IDs unchanged
generated IDs unchanged
decoded text unchanged
selected CPU backend/kernel identity unchanged
dense hook identity unchanged
fallback_used = false
default runtime = eager_f32_candle
packed sidecar scope = layers.0.attention.q_proj.weight only
```

If no safe optimization lands, SLM-CPU-078 should emit a concrete blocker or
next-target artifact rather than weakening the receipt boundary. It must not
enable packed Q8_0 by default, broaden packed sidecar execution to all dense
tensors, claim sustained 8250U throughput, claim broad answer quality, start
Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256 paths.

## SLM-CPU-079 Post-Aligned Matvec Artifact

SLM-CPU-079 captures the first real i5-8250U Qwen3 Q8_0 warm-session artifact
after the SLM-CPU-078 aligned packed Q8_0 exact-tensor matvec implementation.
The committed artifact pack is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-079-post-aligned-matvec-artifact.json
ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-079-post-aligned-matvec-classification.json
```

The run keeps the packed sidecar path opt-in and exact-tensor scoped to
`layers.0.attention.q_proj.weight` / `blk.0.attn_q.weight`. It records
`selected_path = packed_q8_sidecar`, `selected_kernel =
dense-q8-sidecar-linear`, strict GGUF tokenizer authority, `selected_backend =
cpu-rust`, and `fallback_used = false`.

The behavior oracle passes against the SLM-CPU-077 post-bridge sidecar oracle:

```text
prompt_ids_match = true
generated_ids_match = true
decoded_text_match = true
model_sha_match = true
tokenizer_source_match = true
tokenizer_strict_match = true
selected_backend_match = true
fallback_false_before_after = true
```

The post-aligned counter pack records:

```text
selector_dispatch_calls = 42336
selector_selected_calls = 216
selector_declined_calls = 42120
selector_error_calls = 0
input_materialization_calls = 216
bias_materialization_calls = 216
packed_matvec_calls = 216
output_tensor_construction_calls = 216
packed_matvec_ns = 16289575900
```

Compared with the SLM-CPU-077 post-bridge counter pack
(`packed_matvec_ns = 19093586100`), the classification is:

```text
result = improved_bounded_packed_matvec_counter
delta_packed_matvec_ns = -2804010200
packed_matvec_reduction_percent = 14.68561319656971
runtime_promotion_recommended = false
default_runtime_changed = false
speedup_claim = false
```

This is a bounded counter-level classification only. SLM-CPU-079 does not
claim end-to-end speedup, sustained 8250U throughput, broad answer quality,
Q4/Q5 runtime support, server execution, accelerator execution, Qwen3.5
support, or BitNet QK256 changes.

## Current Next Target

The current performance lane has two valid target families. Neither is a
default-runtime promotion gate by itself. After SLM-CPU-090, the
residual-add/output-storage family is blocked at the public Candle Tensor API,
so SLM-CPU-091 selects the non-duplicative `model.logits` / output-head tensor
allocation boundary as the next ready item.

1. Continue the allocation path from the SLM-CPU-035 prefill attribution and
   the SLM-CPU-038 typed transformer-forward workspace boundary by removing or
   narrowly classifying one `prompt_prefill.forward` owned-output allocation.
   The residual-add subpath is currently blocked by SLM-CPU-090's exact Candle
   API finding.
2. Continue the packed Q8_0 exact-tensor path from SLM-CPU-079 by reducing the
   `packed_matvec_compute` counter or by collecting repeated before/after
   warm-session timing receipts, while keeping `packed_q8_sidecar` opt-in and
   exact-tensor scoped.
3. Continue reducing `model.logits` tensor allocation and output-head costs
   without weakening the logits/indexing receipts. SLM-CPU-091 queues this as
   the next non-duplicative implementation or blocker slice.

Both paths must use the Qwen3 Q8_0 appliance profile as the behavior oracle.
Any before/after artifact must preserve:

```text
model SHA
strict GGUF tokenizer authority
prompt IDs
generated IDs
decoded text
selected CPU backend/kernel identity
fallback_used = false
```

If a change improves a counter but changes generated IDs, decoded text, model
identity, tokenizer source, backend/kernel identity, or fallback state, the
change is a failed performance slice. If it helps only one model or one exact
tensor, the dashboard must say so.

This dashboard does not claim end-to-end speedup, sustained 8250U throughput,
broad answer quality, Q4/Q5 runtime support, server execution, GPU/NPU/OpenVINO
or UHD 620 execution, Qwen3.5 support, or BitNet QK256/I2_S changes.

## SLM-CPU-083 Model-Forward Output Boundary

SLM-CPU-083 returns the Kaby Qwen3 Q8_0 performance lane to the
`prompt_prefill.forward` / `model.forward` allocation path after the repeated
packed-Q8 sidecar timing gate regressed. The runtime classification remains
behavior-preserving: `TransformerForwardWorkspace` now records the
`model.forward.output` owned-output surface in addition to the existing
`feed_forward.down_proj.output` surface.

```text
model_forward_owned_output_surface = model.forward.output
model_forward_reuse_status = model_forward_output_storage_blocked_by_owned_tensor_api
workspace_storage_owner = TransformerForwardWorkspace
default_runtime_changed = false
speedup_claim = false
```

The classification narrows the next allocation-layout API boundary: the
current `TransformerModel::forward_with_workspace` path can name the
model-forward output surface, but the behavior-preserving API still returns the
final Candle `Tensor` without caller-provided output storage. Any future reuse
work must add or adopt an output-storage API and must preserve the Qwen3 Q8_0
appliance oracle before claiming even a bounded counter improvement.

This slice does not promote `packed_q8_sidecar`, does not change dense math, and
does not claim end-to-end speedup, sustained 8250U throughput, broad answer
quality, Q4/Q5 runtime support, server execution, accelerator execution,
Qwen3.5 support, or BitNet QK256/I2_S changes.

## SLM-CPU-084 Model-Forward Output Slot

SLM-CPU-084 burns down the `model.forward.output` boundary one step without
changing Qwen3 Q8_0 runtime behavior. `TransformerModel::forward_with_workspace`
now moves the final Candle `Tensor` through a
`TransformerForwardWorkspace`-owned model output slot before returning it to the
caller. This is an explicit behavior-preserving API surface, not reusable output
storage.

```text
model_forward_owned_output_surface = model.forward.output
model_forward_reuse_status = model_forward_output_storage_api_surface_present_reuse_blocked_by_candle_tensor_ops
workspace_storage_owner = TransformerForwardWorkspace
model_workspace_owned_output_count = 1
default_runtime_changed = false
speedup_claim = false
```

Reusable caller-filled output storage remains blocked by the current Candle
owned-tensor operations in the layer loop and final norm path. The next safe
boundary is a final-norm/layer-output caller-output-storage API, still gated by
the Qwen3 Q8_0 appliance oracle: model SHA, strict GGUF tokenizer authority,
prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity,
dense hook identity where applicable, and `fallback_used=false`.

This slice does not promote `packed_q8_sidecar`, does not change dense math, and
does not claim end-to-end speedup, sustained 8250U throughput, broad answer
quality, Q4/Q5 runtime support, server execution, accelerator execution,
Qwen3.5 support, or BitNet QK256/I2_S changes.

## SLM-CPU-085 Final-Norm And Layer-Output Boundary

SLM-CPU-085 narrows the post-`model.forward.output` blocker into explicit
machine-checkable final-norm and block-output surfaces. The workspace records
that both surfaces are reachable without changing generation behavior, but both
still rely on Candle operations that return owned tensors instead of filling
caller-provided storage.

```text
final_norm_output_surface = model.final_norm.output
final_norm_reuse_status = final_norm_output_storage_blocked_by_candle_layer_norm_ops
layer_output_surface = transformer.block.output
layer_output_reuse_status = layer_output_storage_blocked_by_candle_tensor_add_ops
workspace_storage_owner = TransformerForwardWorkspace
default_runtime_changed = false
speedup_claim = false
```

This is a blocker classification, not a speed improvement. Reusable storage
still requires behavior-preserving Candle LayerNorm/RMSNorm and residual-add
output-storage APIs, plus the established Qwen3 Q8_0 appliance oracle, before
any bounded allocation improvement can be claimed.

This slice does not promote `packed_q8_sidecar`, does not change dense math, and
does not claim end-to-end speedup, sustained 8250U throughput, broad answer
quality, Q4/Q5 runtime support, server execution, accelerator execution,
Qwen3.5 support, or BitNet QK256/I2_S changes.

## SLM-CPU-086 Final-Norm Output-Storage Gate

SLM-CPU-086 narrows the `model.final_norm.output` blocker before any
residual-add or dense-math change. The final norm is now classified at the
public Candle norm API boundary: the workspace can identify whether the op is
RMSNorm or LayerNorm, record the epsilon, and prove that input, weight, and
optional bias metadata are readable. The remaining blocker is compute-side:
`LayerNorm::forward` and the public `candle_nn::ops` norm helpers still return
owned tensors and do not accept caller-provided output storage.

```text
final_norm_output_surface = model.final_norm.output
final_norm_operation_detail = rms_norm
final_norm_caller_output_helper_status = final_norm_output_storage_helper_blocked_by_owned_candle_norm_output
post_model_forward_required_api_boundary = final_norm_output_storage_api_or_apply_op_output_hook
can_fill_final_norm_output_storage = false
default_runtime_changed = false
speedup_claim = false
```

This is a narrower blocker than SLM-CPU-085, not a runtime improvement. A later
slice must add or adopt a behavior-preserving Candle LayerNorm/RMSNorm
caller-output-storage API, then regenerate before/after receipts proving the
same model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs,
decoded text, selected CPU backend/kernel identity, and `fallback_used=false`
before claiming even a bounded allocation improvement.

This slice does not move to residual-add/layer-output runtime changes, promote
`packed_q8_sidecar`, claim speedup, claim sustained 8250U throughput, broaden
Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet
QK256/I2_S paths.

## SLM-CPU-088 Residual Block-Output Boundary

SLM-CPU-088 moves the active allocation-layout blocker from the final-norm API
surface to the residual-add / `transformer.block.output` boundary queued by
SLM-CPU-087. The typed workspace now records the two owned tensors that feed the
second block residual add: the post-attention residual input and the
feed-forward branch output. That makes the remaining blocker machine-checkable:
both input shapes are known, the output shape is known, and the operation family
is the Candle tensor residual-add path that still returns an owned `Tensor`
instead of filling caller-provided storage.

```text
layer_output_surface = transformer.block.output
layer_output_operation_family = candle_core::Tensor residual_add
layer_output_operation_detail = residual_add_owned_tensor_output
layer_output_residual_input_shape_recorded = true
layer_output_branch_output_shape_recorded = true
layer_output_caller_output_helper_status = layer_output_storage_helper_blocked_by_owned_candle_residual_add_output
can_fill_layer_output_storage = false
next_safe_change = residual-add caller-output-storage API
default_runtime_changed = false
speedup_claim = false
```

This is still a blocker/implementation-boundary slice. It does not replace the
residual add, does not reuse tensor storage, and does not claim a runtime
improvement. A later optimization must add or adopt a behavior-preserving Candle
Tensor residual-add caller-output-storage API and then prove the Qwen3 Q8_0
appliance oracle again with matching model SHA, strict GGUF tokenizer authority,
prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity,
dense hook identity where applicable, and `fallback_used=false`.

This slice does not promote `packed_q8_sidecar`, does not change dense math, and
does not claim end-to-end speedup, sustained 8250U throughput, broad answer
quality, Q4/Q5 runtime support, server execution, accelerator execution,
Qwen3.5 support, or BitNet QK256/I2_S changes.

## SLM-CPU-089 Residual-Add Storage Gate

SLM-CPU-089 is the next queued performance-lane gate after SLM-CPU-088. It does
not start from a speed target. It starts from the specific blocker that
SLM-CPU-088 made machine-checkable:

```text
blocked_surface = transformer.block.output
blocked_operation = residual_add
current_output_ownership = owned Candle Tensor
required_next_boundary = behavior-preserving caller-provided output storage
default_runtime_changed = false
speedup_claim = false
```

A valid implementation may add a narrow residual-add output-storage API or may
prove that the current Candle tensor boundary still prevents reusable storage.
Either result must stay useful to the performance lane by preserving the exact
claim boundary: no runtime improvement can be claimed until before/after Qwen3
Q8_0 receipts prove matching model SHA, strict GGUF tokenizer authority, prompt
IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense
hook identity where applicable, and `fallback_used=false`.

This is intentionally still in the allocation/layout lane. It must not promote
`packed_q8_sidecar`, broaden Q4/Q5 support, claim sustained 8250U throughput,
or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S
paths.

## SLM-CPU-090 Residual-Add Output-Storage Slice

SLM-CPU-090 is the next non-duplicative step after the SLM-CPU-089 queue gate.
It is not allowed to infer performance from intent. It must either implement a
behavior-preserving residual-add helper for `transformer.block.output` that can
fill caller-provided reusable storage, or produce a machine-checkable blocker
that names the exact Candle tensor API limitation preventing that storage reuse.

```text
target_surface = transformer.block.output
target_operation = residual_add
current_blocker = Candle residual-add returns owned Tensor output
allowed_result = reusable-storage helper or exact API blocker
required_behavior_oracle = Qwen3 Q8_0 appliance receipt equivalence
default_runtime_changed_without_oracle = false
speedup_claim_without_before_after_receipts = false
```

The SLM-CPU-090 blocker must be exact enough to be machine-checkable. The
current public Candle residual-add boundary is:

```text
blocking_ops =
  Tensor::add(&self, &Tensor) -> Result<Tensor>
  Tensor::broadcast_add(&self, &Tensor) -> Result<Tensor>
  std::ops::Add for Tensor/&Tensor delegates to Tensor::add and returns Result<Tensor>
public_api_accepts_output_storage = false
backend_internal_in_place_api_exposed = false
required_missing_api = add_out/broadcast_add_out or equivalent Tensor residual-add API accepting caller-provided output storage
```

If SLM-CPU-090 changes runtime behavior, the change must be paired with
before/after Qwen3 Q8_0 appliance evidence showing unchanged model SHA, strict
GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU
backend/kernel identity, dense hook identity where applicable, and
`fallback_used=false`. If the API is still blocked, the blocker must be concrete
enough to guide the next Candle/output-storage work rather than restating the
generic owned-output problem.

This slice must not promote `packed_q8_sidecar`, claim end-to-end speedup,
claim sustained 8250U throughput, broaden Q4/Q5 support, or touch server, GPU,
NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-091 Logits / Output-Head Boundary

SLM-CPU-091 is the next queued performance-lane target after SLM-CPU-090. The
residual-add reusable-storage path is now blocked by a concrete public Candle
Tensor API limitation:

```text
Tensor::add(&self, &Tensor) -> Result<Tensor>
Tensor::broadcast_add(&self, &Tensor) -> Result<Tensor>
std::ops::Add delegates to Tensor::add
public_api_accepts_output_storage = false
backend_internal_in_place_api_exposed = false
```

Rather than re-stating that blocker, SLM-CPU-091 moves to the next remaining
allocation surface that the dashboard already names:

```text
target_surface = model.logits / output-head tensor allocation
current_boundary = logits/output-head owned tensor or extraction path
allowed_result = behavior-preserving reduction or exact blocker classification
required_behavior_oracle = Qwen3 Q8_0 appliance receipt equivalence
default_runtime_changed_without_oracle = false
speedup_claim_without_before_after_receipts = false
```

Any runtime change must preserve model SHA, strict GGUF tokenizer authority,
prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity,
dense hook identity where applicable, and `fallback_used=false`. If the slice
only narrows a blocker, it must make the blocker concrete enough to guide the
next implementation step.

This slice must not claim residual-add storage is solved, promote
`packed_q8_sidecar`, claim end-to-end speedup, claim sustained 8250U
throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD
620, Qwen3.5, or BitNet QK256/I2_S paths.

SLM-CPU-091 records the boundary in the warm-session allocation audit rather
than changing runtime selection:

```text
status = logits_output_storage_blocked_by_candle_tensor_ops
exact_blocking_ops =
  candle_nn::Linear::forward(&self, &Tensor) -> Result<Tensor>
  Tensor::matmul(&self, &Tensor) -> Result<Tensor>
  Tensor::reshape(&self, shape) -> Result<Tensor>
  Tensor::to_vec1::<f32>(&self) -> Result<Vec<f32>> when host logits extraction is requested
required_missing_api =
  logits/output-head API accepting caller-provided output storage
  or a fused top-k/argmax path that avoids materializing a full owned logits tensor
```

This preserves the earlier sampler/logits-scratch cleanup while making the
remaining owned-output surface explicit. It does not alter generated IDs,
decoded text, tokenizer authority, backend identity, dense hook identity,
packed-Q8 selection, or thread defaults.

## SLM-CPU-096 Runtime Output Tensor Storage Gate

SLM-CPU-096 follows the SLM-CPU-095 caller-output-slice helper gate. The inner
packed-Q8 matvec helpers can fill caller-provided `&mut [f32]` output slices,
but the full runtime still has to return a Candle `Tensor` to preserve the
existing dense-linear interface:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-096-runtime-output-tensor-storage-gate.json
previous_artifact = ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-095-packed-q8-matvec-output-scratch.json
current_runtime_construction = Tensor::from_vec(output, output_shape, input.device())
public_alternative_reviewed = Tensor::from_storage(Storage::Cpu(CpuStorage::F32(output)), output_shape, input.device())
classification.status = blocked_by_candle_owned_storage_api
runtime_storage_reuse_supported = false
safe_runtime_change_available_now = false
```

Both reviewed public construction paths transfer owned `Vec<f32>` storage into
the returned Candle `Tensor`. That preserves returned-tensor semantics, but it
does not let the packed-Q8 runtime keep reusable caller-owned output storage
alive after Tensor construction. The next safe direction is either a Candle API
with explicit caller-owned output-storage lifetime semantics, or a fused
consumer path that avoids returning a Candle Tensor at this boundary and proves
Qwen3 Q8_0 behavior equivalence with before/after receipts.

This slice does not change the default runtime, promote `packed_q8_sidecar`,
claim allocation improvement, claim speedup, claim sustained 8250U throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256/I2_S paths.

## SLM-CPU-099 Typed Fused Q Consumer Contract

SLM-CPU-099 follows the SLM-CPU-098 fused-consumer boundary classification. It
does not implement a fused runtime path. It defines the narrow contract a future
implementation must satisfy before it may avoid the returned Candle `Tensor` at
the exact Q projection boundary:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-099-typed-fused-q-consumer-contract.json
exact_tensor = layers.0.attention.q_proj.weight
role = attention.q_proj.typed_fused_consumer_contract
status = contract_defined_runtime_disabled
runtime_execution_enabled = false
intermediate_returned_candle_tensor_allowed = false
```

The contract orders the required downstream stages:

```text
packed_q8_matvec_output_slice
q_proj_reshape
q_proj_transpose
optional_q_norm
q_rope
trace_workspace_identity
attention_head_handoff
```

Any later runtime implementation remains gated by repeated before/after receipts
that preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated
IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, and
`fallback_used=false`. This slice does not change the default runtime, promote
`packed_q8_sidecar`, prove allocation reduction, claim speedup, claim sustained
8250U throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO,
UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-081 Repeated Timing Gate

SLM-CPU-081 records the next evidence boundary for the exact-tensor packed-Q8
path after the SLM-CPU-079 counter improvement:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-22/qwen3-slm-cpu-081-repeated-packed-q8-timing-gate.json
classification.result = not_claimed
minimum_receipts_per_side = 3
current_baseline_receipts = 1
current_candidate_receipts = 1
```

The available evidence still matters: the SLM-CPU-079 candidate preserved the
Qwen3 Q8_0 behavior oracle and reduced the bounded `packed_matvec_ns` counter
relative to the SLM-CPU-077 sidecar oracle. That is a counter-level result, not
repeated end-to-end timing evidence. A future timing classification must compare
at least three behavior-equivalent baseline receipts and three
behavior-equivalent candidate receipts for the same host, model SHA, thread
count, prompt corpus, tokenizer authority, backend identity, dense hook identity,
and `fallback_used = false` boundary.

The exact-tensor packed-Q8 sidecar remains opt-in, scoped to
`layers.0.attention.q_proj.weight`, and is not promoted to the default runtime by
this gate.

## SLM-CPU-082 Repeated Receipt Capture

SLM-CPU-082 captures the repeated i5-8250U Qwen3 Q8_0 warm-session receipt pack
required by the SLM-CPU-081 timing gate:

```text
baseline receipts = 3
candidate receipts = 3
baseline selected_path = eager_f32_candle
candidate selected_path = packed_q8_sidecar
candidate exact tensor = layers.0.attention.q_proj.weight
classification.result = regressed
classification.prompt_total_ratio = 1.11186
behavior_equivalence.passed = true
speedup_claim = false
sustained_throughput_claim = false
```

The receipts preserve the Qwen3 Q8_0 behavior oracle: strict GGUF tokenizer
authority, selected CPU backend, `fallback_used = false`, deterministic corpus
quality, generated IDs, and decoded text. The timing classification is bounded
to this host, model, corpus, 4-thread setting, and exact-tensor opt-in sidecar.
It does not enable packed-Q8 by default, generalize beyond the exact Q
projection tensor, or claim sustained throughput.

## SLM-CPU-100 Typed Fused Q Consumer Implementation Gate

SLM-CPU-100 attempts the implementation gate defined by SLM-CPU-099 and records
the current blocker rather than enabling a speculative runtime path. The inner
packed-Q8 matvec helpers can already write into caller-owned output slices, but
the exact `layers.0.attention.q_proj.weight` consumer still requires Candle
`Tensor` semantics immediately after projection:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-24/qwen3-slm-cpu-100-typed-fused-q-consumer-implementation-gate.json
exact_tensor = layers.0.attention.q_proj.weight
role = attention.q_proj.typed_fused_consumer_implementation_gate
status = blocked_runtime_disabled
runtime_execution_enabled = false
default_runtime_changed = false
```

The machine-checkable blockers are:

```text
q_heads_tensor_semantics
q_norm_tensor_api
rope_tensor_api
trace_workspace_tensor_identity
attention_handoff_tensor_contract
receipt_safety_evidence
```

The next safe slice is a typed attention-head buffer/view that can carry the Q
projection through reshape, transpose, optional q_norm, RoPE, trace/workspace
identity, and score handoff without constructing an intermediate returned
Candle `Tensor`. Runtime execution still requires before/after Qwen3 Q8_0
receipts proving identical model SHA, strict GGUF tokenizer authority, prompt
IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense
hook identity, and `fallback_used=false`.

This slice does not change the default runtime, promote `packed_q8_sidecar`,
prove allocation reduction, claim speedup, claim sustained 8250U throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256/I2_S paths.

SLM-CPU-101 defines that typed attention-head view as a runtime-disabled
contract. The exact Q projection can be represented as a logical
`[batch, n_heads, seq, head_dim]` view over packed-Q8 matvec output storage
without a returned intermediate Candle `Tensor`, but current downstream
consumers still require Tensor-shaped APIs:

```text
status = contract_defined_runtime_disabled
can_represent_q_heads_without_candle_tensor = true
can_feed_current_attention_score_api_without_materialization = false
selected_materialization_point = null
runtime_execution_enabled = false
default_runtime_changed = false
allocation_reduction_claim = false
speedup_claim = false
```

The machine-checkable blockers are:

```text
q_norm_requires_tensor_or_typed_norm
rope_requires_tensor_or_typed_rope
trace_source_identity_requires_tensor_mapping
attention_scores_require_tensor_or_typed_score_path
receipt_safety_evidence
```

The next safe slice is a typed q_norm/RoPE path or an explicit
single-materialization score-handoff boundary, still gated by repeated
before/after Qwen3 Q8_0 receipts proving identical model SHA, strict GGUF
tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU
backend/kernel identity, dense hook identity, and `fallback_used=false`.

This slice does not change the default runtime, promote `packed_q8_sidecar`,
prove allocation reduction, claim speedup, claim sustained 8250U throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256/I2_S paths.
