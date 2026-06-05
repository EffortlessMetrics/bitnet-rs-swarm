# Kaby Lake SLM CPU Performance Dashboard

This dashboard is the baseline for i5-8250U dense SLM performance work. It
summarizes committed strict CPU receipts for the Qwen3-0.6B Q8_0 appliance
profile and the bounded Qwen2.5-0.5B Q8_0 second-model sanity path. It is not a
sustained throughput claim and it does not broaden support to Q4/Q5, server,
GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256.

## Evidence Set

| Evidence | Path | Role |
| --- | --- | --- |
| SLM-CPU-205 machine-readable dashboard evidence | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-205-kaby-performance-dashboard-evidence.json` | Consolidates committed Qwen3 and Qwen2.5 i5-8250U timing, memory, storage, thermal/power, allocation, and thread-count evidence without changing runtime behavior |
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
| q_proj output proof refresh | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-122-qproj-output-proof-refresh.json` | Refreshes the shared q_proj-output before/after proof after SLM-CPU-121; Qwen3 behavior and fingerprint now match, but Qwen2.5 still preserves generated behavior while changing the f32-le q_proj-output fingerprint, so the boundary remains fail-closed |
| Qwen2.5 q_proj fingerprint root cause | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen25-slm-cpu-123-qproj-fingerprint-root-cause.json` | Classifies the remaining Qwen2.5 blocker as a sidecar-gated q_proj-output equivalence gap that cannot be numerically localized from fingerprint-only traces; it keeps packed-Q8 sidecar behavior proof and performance claims fail-closed until a bounded tensor sample or full 896-f32 dump exists |
| Qwen2.5 q_proj tensor dump classification | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen25-slm-cpu-124-qproj-tensor-dump-classification.json` | Captures an opt-in full 896-f32 q_proj-output diagnostic dump for the Qwen2.5 before/after pair and classifies the fingerprint delta as small f32 numeric drift while generated behavior remains stable; packed-Q8 sidecar behavior proof, allocation, timing, throughput, and default-runtime promotion remain fail-closed |
| q_proj numeric tolerance gate | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-125-qproj-numeric-tolerance-gate.json` | Accepts a narrow absolute `1e-4` f32 gate for the exact layer-0 `attention.q_proj_output_pre_optional_qnorm` diagnostic boundary using the accepted Qwen3 exact-match evidence and Qwen2.5 full-vector bounded-drift evidence; this proves only that exact boundary and does not claim allocation reduction, timing improvement, throughput, or default-runtime promotion |
| q_proj allocation/timing readiness blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-126-allocation-timing-readiness-blocker.json` | Keeps the next before/after allocation or timing experiment fail-closed because the fresh cross-model receipt prerequisites are incomplete in this workspace: Qwen3 Q8_0 is present, but the exact Qwen2.5 Q8_0 GGUF needed for fresh before/after receipts is missing |
| q_proj fresh receipt prerequisites | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-127-fresh-receipt-prereq.json` | Restores the exact Qwen2.5 Q8_0 cache artifact by pinned SHA256 and records fresh Qwen3/Qwen2.5 before/after strict CPU receipts with stable generated IDs/text, `cpu-rust`, `dense-qwen-cpu-reference`, and `fallback_used=false`; this is receipt readiness only, not an allocation, timing, speedup, or default-runtime claim |
| q_proj timing classification | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-128-qproj-timing-classification.json` | Consumes the SLM-CPU-127 receipt pack and classifies the exact-tensor packed-Q8 sidecar timing evidence as mixed: Qwen3 is neutral on one sample, Qwen2.5 regresses, and allocation-audit counters are absent; no speedup, timing improvement, allocation reduction, or default-runtime promotion is claimed |
| Repeated q_proj allocation audit | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-129-repeated-allocation-audit.json` | Collects repeated warm-session allocation-audit receipts for Qwen3 and Qwen2.5 after SLM-CPU-128; generated IDs/text remain behavior-equivalent and counters are available, but Qwen3 does not select packed sidecar compute and Qwen2.5 remains opt-in/counter-scoped, so no allocation reduction, speedup, timing improvement, or default-runtime promotion is claimed |
| q_proj selector convergence gate | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-130-selector-convergence-gate.json` | Explains the cross-model selector mismatch from SLM-CPU-129: Qwen3 is correctly declined by the payload-order guard because `sidecar_payload_order_matches_runtime_shape=false`, while Qwen2.5 reaches the opt-in packed-Q8 counter path with matching payload order; the guard remains fail-closed and no runtime promotion or performance claim is made |
| Qwen3 q_proj payload-order proof | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-slm-cpu-131-qproj-payload-order-proof.json` | Records the machine-checkable Qwen3 blocker: the GGUF source shape `[1024, 2048]` maps to Candle runtime shape `[2048, 1024]`, so the packed Q8_0 source payload must remain fail-closed until a tensor-specific reorder/runtime-shape proof exists; no selector relaxation or performance claim is made |
| Qwen3 q_proj payload-reorder contract | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-slm-cpu-132-qproj-payload-reorder-contract.json` | Defines the fail-closed reorder contract for the same exact tensor: a pure Q8_0 byte reorder is not valid because transposing from source shape `[1024, 2048]` to runtime shape `[2048, 1024]` would regroup values under the wrong 32-value Q8_0 block scales; runtime selection remains blocked until a dequantize/requantize proof or source-order Q8_0 kernel exists |
| Qwen3 source-order q_proj kernel contract | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-slm-cpu-133-source-order-qproj-kernel-contract.json` | Defines the runtime-disabled source-order Q8_0 matvec contract for the same tensor: consume GGUF source-order rows directly with an output accumulator rather than byte-transposing or dequantize/requantizing the payload; selection remains disabled until accumulator order, block-scale decode, generated-ID preservation, and receipt identity are proven |
| Qwen3 source-order q_proj matvec prototype | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-slm-cpu-134-source-order-qproj-matvec-prototype.json` | Adds a runtime-disabled source-order Q8_0 matvec prototype surface that accumulates GGUF source-order rows into the runtime output vector and proves the fixture implementation against an eager source-order f32 reference; selector use remains blocked until exact Qwen3/Qwen2.5 behavior receipts preserve generated IDs and receipt identity |
| Source-order q_proj behavior gate | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-135-source-order-qproj-behavior-gate.json` | Defines the exact-model receipt gate required before the source-order q_proj matvec prototype can be considered for selector use; Qwen3 and Qwen2.5 before/after receipts for this selector path are missing, so runtime selection remains disabled |
| Source-order q_proj receipt-pair blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-136-source-order-qproj-receipt-pair-blocker.json` | Blocks source-order q_proj selector receipt capture because the generation-time selector hook, candidate receipt identity, and runtime input binding are not wired; runtime selection remains disabled until those surfaces exist and Qwen3/Qwen2.5 before/after receipts preserve generated IDs and receipt identity |
| Source-order q_proj selector identity gate | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-137-source-order-qproj-selector-hook.json` | Adds the default-disabled source-order q_proj candidate identity surface to strict receipts while keeping eager F32 as the default runtime |
| Source-order q_proj receipt-pair blocker refresh | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-138-source-order-qproj-receipt-pair-blocker.json` | Keeps source-order selector use fail-closed because this workspace has Qwen3 Q8_0 only, the exact Qwen2.5 Q8_0 model cache is missing, and the existing cross-model receipts predate the SLM-CPU-137 source-order identity fields |
| Source-order exact-model blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-139-source-order-exact-model-receipt-blocker.json` | Re-checks the current-main exact-model source-order receipt gate and blocks because the exact Qwen2.5 Q8_0 cache is still absent and the older receipt pairs predate the source-order identity fields |
| Qwen2.5 cache restore contract | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-140-cache-restore-contract.json` | Defines the non-committed Qwen2.5 Q8_0 cache restoration and Qwen3/Qwen2.5 source-order receipt-capture contract; it captures no fresh receipts and makes no runtime, allocation, timing, or speed claim |
| Source-order q_proj receipt capture | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-141-source-order-receipt-capture.json` | Restores and verifies the exact Qwen2.5 Q8_0 cache, captures fresh Qwen3/Qwen2.5 before/after strict CPU warm-session receipts, and keeps selector use blocked because q_proj numeric evidence is absent and Qwen2.5 does not record the source-order q_proj candidate identity |
| Source-order q_proj evidence fields | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-142-source-order-evidence-fields.json` | Adds explicit fail-closed receipt fields for source-order q_proj candidate identity and missing q_proj numeric evidence, resolving absent-field ambiguity without runtime promotion or performance claims |
| Source-order q_proj evidence receipt refresh | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-143-source-order-evidence-receipts.json` | Captures fresh current-main Qwen3/Qwen2.5 before/after strict CPU warm-session receipts with the SLM-CPU-142 fields present; generated behavior is preserved, but selector use remains blocked because source-order q_proj candidate identity is still not usable and q_proj numeric evidence is not captured |
| Source-order q_proj identity blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-144-source-order-qproj-identity-blocker.json` | Classifies the SLM-CPU-143 source-order identity gap as a missing payload-gate capture plus a receipt-classifier null-boundary ambiguity; runtime selection remains disabled |
| Payload/runtime q_proj capture | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-145-payload-runtime-qproj-capture.json` | Captures exact Qwen3/Qwen2.5 payload+runtime gated receipts for `blk.0.attn_q.weight`: Qwen3 exposes the source-order q_proj identity but remains fail-closed on payload-order mismatch, while Qwen2.5 preserves generated IDs/text with the opt-in packed q_proj path selected; selector promotion remains blocked because q_proj numeric evidence is not attached |
| Payload/runtime q_proj numeric gate | `ci/slm-cpu/intel-i5-8250u/2026-05-26/qwen3-qwen25-slm-cpu-146-qproj-numeric-evidence-gate.json` | Consumes the SLM-CPU-145 payload/runtime receipt oracles and precisely blocks selector promotion because the warm-session receipt surface still lacks attached q_proj f32 fingerprint/vector diff evidence |
| Warm-session q_proj capture surface | `ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-148-warm-session-qproj-numeric-evidence.json` | Consumes the SLM-CPU-147 trace surface: Qwen2.5 captures accepted before/after q_proj numeric evidence under explicit packed-Q8 gates, while Qwen3 preserves eager q_proj behavior and remains blocked from runtime-sidecar evidence at the payload-order/runtime-shape boundary |
| Qwen3 source-order runtime boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-slm-cpu-149-source-order-qproj-runtime-boundary.json` | Precisely blocks Qwen3 source-order q_proj runtime evidence because current generation receipts expose candidate identity and eager q_proj numeric preservation, but no receipt-safe generation-time source-order input binding or candidate output vector exists |
| Source-order mapped q_proj candidate | `ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-155-source-order-mapped-qproj-candidate.json` | Applies the SLM-CPU-154 runtime row mapping to the default-disabled Qwen3 source-order q_proj candidate; the trace-gated candidate matches the eager q_proj oracle within tolerance, while `eager_f32_candle` remains the default runtime |
| Source-order q_proj selector gate | `ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-156-source-order-qproj-selector-gate.json` | Precisely blocks source-order q_proj selector promotion after SLM-CPU-155 because paired Qwen3/Qwen2.5 strict CPU before/after behavior receipts with q_proj numeric evidence are still required; `eager_f32_candle` remains the default runtime |
| Source-order q_proj receipt pair | `ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-157-source-order-qproj-receipt-pair.json` | Captures paired Qwen3/Qwen2.5 strict CPU before/after receipts with prompt IDs, generated IDs/text, backend/kernel identity, fallback=false, and q_proj numeric evidence; Qwen3 source-order candidate remains runtime-disabled/default-disabled and Qwen2.5 opt-in sidecar stays within the accepted `1e-4` q_proj tolerance, with no runtime promotion or performance claim |
| Residual-add storage API decision | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-170-residual-add-storage-api-decision.json` | Consumes the SLM-CPU-169 frontier metadata and records the residual-add output-storage path as blocked until Candle exposes an `add_out` / `broadcast_add_out`-style API or a verified backend-local equivalent; no runtime allocation behavior changes or performance claims are made |
| Post-residual allocation frontier | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-171-post-residual-allocation-frontier.json` | Selects `prompt_tokenize` as the next measured Qwen3/Qwen2.5 allocation frontier not blocked by Candle residual-add caller-output-storage support; no runtime allocation behavior changes or performance claims are made |
| Prompt-tokenize allocation contract | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-172-prompt-tokenize-allocation-contract.json` | Defines the prompt identity, tokenizer provenance, rendered-prompt, and prompt-ID cache contract required before any prompt-tokenize runtime reuse or allocation claim |
| Prompt-tokenize cache evidence fields | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-173-prompt-tokenize-cache-evidence-fields.json` | Makes the SLM-CPU-172 prompt-tokenize cache contract receipt-visible without changing prompt tokenization behavior or claiming allocation/timing improvement |
| Prompt-tokenize paired receipt gate | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-174-prompt-tokenize-paired-receipt-gate.json` | Defines the exact Qwen3/Qwen2.5 before/after receipt gate required before any prompt-tokenize allocation, latency, or timing claim |
| Prompt-tokenize exact-identity cache | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-175-prompt-tokenize-exact-identity-cache.json` | Seeds the resident warm-session exact-identity prompt-token cache during pre-sizing so byte-identical prompt-loop lookups hit for both Qwen3 and Qwen2.5, with paired strict receipts preserving model SHA, strict GGUF tokenizer authority, prompt/generated IDs, decoded text, CPU backend identity, and fallback=false |
| Post-prompt-tokenize frontier selection | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-176-post-prompt-tokenize-frontier.json` | Consumes the SLM-CPU-175 receipts and selects `model_forward_owned_tensor_allocation_boundary` as the next high-value frontier, while deferring small prompt setup evidence allocations and keeping SLM-CPU-175 claims scoped to byte-identical cache hits |
| Model-forward allocation boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-177-model-forward-allocation-boundary.json` | Classifies the selected `model_forward_owned_tensor_allocation_boundary` as blocked by dense-linear, final-norm, residual-add, and model.forward owned Candle Tensor outputs, and selects dense-linear output-storage feasibility as the next sub-boundary without changing runtime behavior |
| Dense-linear output-storage feasibility | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-178-dense-linear-output-storage-feasibility.json` | Records that dense-linear weights and optional bias are readable, but production compute still returns owned Candle Tensor outputs; backend-local host slices cannot cross the returned-Tensor boundary, so a later runtime gate needs matmul/bias output-storage APIs or a fully typed fused consumer |
| Dense-linear caller-output-storage gate | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-179-dense-linear-caller-output-storage-runtime-gate.json` | Defines the disabled gate, API capabilities, cross-model receipt requirements, and failure policy required before dense-linear caller-owned output storage can be enabled; runtime behavior remains unchanged |
| Fused dense-consumer feasibility | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-180-fused-dense-consumer-feasibility.json` | Classifies a narrow repo-owned fused dense-linear consumer as too broad because it would need to own residual add, trace/workspace identity, block output identity, and next norm/model-forward Tensor semantics; runtime behavior remains unchanged |
| No-bias dense-linear frontier | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-182-no-bias-dense-linear-frontier.json` | Classifies no-bias dense-linear as a future disabled gate: existing strict Qwen3/Qwen2.5 receipts show zero sidecar bias materialization calls, but committed evidence does not prove every dense-linear role is biasless, and the no-bias path does not resolve the Candle owned-output Tensor boundary |
| Dense bias manifest gate | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-183-dense-bias-manifest-gate.json` | Defines the fail-closed per-role bias-presence manifest or model-init trace required before any future no-bias dense-linear fast path; unknown or present bias blocks selection and runtime behavior remains unchanged |
| Dense bias manifest capture blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-184-dense-bias-manifest-capture-blocker.json` | Blocks manifest capture from committed evidence because model-init trace code and aggregate bias counters are not a complete per-role Qwen3/Qwen2.5 bias-presence manifest; no runtime selection changes |
| Model-init bias manifest trace export | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-185-model-init-bias-manifest-trace-export.json` | Adds the missing trace export coverage for feed-forward and output-head dense-linear bias presence so future exact-model traces can derive per-role/layer `role_records`; runtime selection and no-bias fast paths remain disabled |
| Dense bias manifest trace capture blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-186-dense-bias-manifest-trace-capture-blocker.json` | Reviews committed pre-export traces, records Qwen3 attention bias absence and Qwen2.5 attention q/k/v bias presence, and blocks complete manifest capture until fresh post-export Qwen3/Qwen2.5 traces cover feed-forward and output-head roles |
| Post-export dense bias trace capture | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-188-post-export-bias-trace-capture.json` | Captures post-export `model_init.linear_bias_finish` traces and paired strict warm-session receipts for Qwen3/Qwen2.5 Q8_0; Qwen3 records all selected dense-linear roles biasless, while Qwen2.5 records attention q/k/v bias present, keeping any blanket no-bias fast path fail-closed |
| Dense bias role records manifest | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-189-dense-bias-role-records-manifest.json` | Derives 366 exact post-export role records from the captured Qwen3/Qwen2.5 traces and paired strict receipts; no runtime selector is enabled, and Qwen2.5 attention q/k/v bias presence blocks blanket no-bias selection |
| No-bias selector policy gate | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-190-no-bias-selector-policy-gate.json` | Defines a runtime-disabled, fail-closed selector policy from the role records manifest: 294 biasless role records are eligible candidates, 72 biased Qwen2.5 attention q/k/v records are blocked, and no runtime path is selected |
| No-bias selector dry run | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-191-no-bias-selector-dry-run-receipts.json` | Applies the disabled policy to all 366 role records and emits receipt-visible decisions: 294 roles are eligible future no-bias candidates, 72 Qwen2.5 attention q/k/v roles fail closed because bias is present, and runtime selection remains disabled |
| No-bias selector audit hook | `ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-192-no-bias-selector-audit-hook.json` | Adds a typed audit-only `DenseLinearNoBiasSelectorAudit` boundary: biasless roles are future candidates, biased or unknown roles fail closed, and the selected path remains `eager_f32_candle` / `dense-f32-candle-linear` |
| No-bias apply-linear receipt capture blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-207-no-bias-apply-linear-receipt-capture-blocker.json` | Verifies the explicit Qwen3 and Qwen2.5 Q8_0 model paths are present by SHA, but blocks fresh before/after capture because `slm-warm-session` currently passes `None` into the no-bias apply-linear gate emitter; long receipt runs would not populate the required descriptor/callsite or digest-bound gate fields |
| No-bias apply-linear gate wiring | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-208-no-bias-apply-linear-gate-wiring.json` | Wires a real digest-bound no-bias apply-linear gate object into eligible Qwen3/Qwen2.5 Q8_0 warm-session aggregates while keeping candidate execution disabled and blocking on fresh before/after receipts |
| No-bias apply-linear receipt pairs | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-209-no-bias-apply-linear-receipt-pairs.json` | Captures fresh explicit-path Qwen3/Qwen2.5 Q8_0 before/after strict warm-session receipts through the wired gate, preserving model/tokenizer/runtime/path/kernel/digest identity while candidate execution and normal runtime selection remain disabled |
| No-bias candidate off/on receipt gate | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-215-no-bias-candidate-off-on-receipt-gate.json` | Defines the exact candidate-off/candidate-on receipt-pair gate for Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj`; the gate remains fail-closed because no candidate-on strict warm-session receipt exists |
| No-bias candidate-on behavior evidence gate | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-216-no-bias-candidate-on-behavior-evidence-gate.json` | Defines the fail-closed behavior-evidence boundary after the receipt-pair gate; candidate execution remains disabled because `FeedForward::apply_linear` lacks a candidate-on runtime attachment point and complete strict receipt fields |
| No-bias candidate runtime attachment boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-217-no-bias-candidate-runtime-attachment.json` | Defines the explicit candidate-on apply-linear runtime attachment boundary and records the remaining blocker as the missing candidate runtime owner/receipt emitter; eager_f32_candle remains the default runtime |
| No-bias runtime attempt blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-229-no-bias-runtime-attempt-blocker.json` | Consumes the validated SLM-CPU-228 strict capture pair and records that candidate execution remains blocked until receipt-bound selector identity reaches the dense runtime hook registry and apply-linear dispatch has a separately gated execution receipt |
| No-bias runtime hook attachment | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-230-no-bias-runtime-hook-attachment.json` | Proves the receipt-bound no-bias selector can be attached to `DenseLinearRuntimeHookRegistry` for Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj` while preserving eager_f32_candle and keeping candidate execution disabled |
| No-bias candidate execution receipt gate | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-231-no-bias-candidate-execution-receipt-gate.json` | Consumes the SLM-CPU-230 runtime hook attachment and keeps candidate execution fail-closed until fresh candidate-off/candidate-on execution receipts prove generated IDs, decoded text, backend/kernel identity, model SHA, tokenizer authority, and fallback=false are preserved |
| No-bias execution capture commands | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-232-no-bias-execution-capture-commands.json` | Defines the concrete candidate-off/candidate-on execution capture command contract for Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj`; receipts remain uncaptured and no candidate execution or preservation claim is made |
| No-bias execution receipt blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-233-no-bias-execution-receipt-blocker.json` | Blocks fresh SLM-CPU-233 candidate-off/candidate-on execution receipt capture because the pinned Qwen2.5 Q8_0 GGUF is absent from this workspace; Qwen3 is present and SHA-verified, but no candidate execution receipt or preservation claim is made |
| Qwen2.5 artifact prerequisite | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-234-qwen25-artifact-prereq.json` | Verifies the pinned Qwen2.5 Q8_0 GGUF from ignored local `target` caches by SHA without committing a model binary, clearing the artifact prerequisite for a later fresh no-bias candidate-off/candidate-on capture |
| No-bias execution receipt capture | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-235-no-bias-execution-capture-validation.json` | Captures fresh Qwen3/Qwen2.5 Q8_0 explicit gate-off/gate-on warm-session receipts for `feed_forward.down_proj`; generated IDs and decoded text are preserved while candidate execution remains disabled and `eager_f32_candle` remains selected |
| No-bias candidate execution attempt boundary | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-236-no-bias-candidate-execution-attempt.json` | Consumes the validated SLM-CPU-235 receipt pair and blocks candidate execution at the exact current runtime boundary: `FeedForward::apply_linear` has no no-bias candidate dispatch branch, so eager F32 remains selected and no executed-candidate preservation or speed claim is made |
| No-bias apply-linear dispatch blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-237-no-bias-dispatch-blocker.json` | Refines the SLM-CPU-236 blocker: a dispatch branch would be unreachable or receipt-unsafe until prompt-bound no-bias selector identity can reach `FeedForward::apply_linear`; production hook registry construction still sets `receipt_bound_no_bias_selector=None` |
| No-bias per-callsite receipt emitter | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-238-per-callsite-no-bias-receipt-emitter.json` | Defines the safer per-callsite receipt-emitter boundary that avoids mutating model-load hooks while candidate execution remains disabled |
| No-bias per-callsite off/on blocker | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-239-per-callsite-no-bias-off-on-receipts.json` | Records that existing off/on receipts are request-gate evidence only because the candidate-on path still does not execute from `FeedForward::apply_linear` |
| No-bias per-callsite dispatch descriptor | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-240-per-callsite-no-bias-dispatch-descriptor.json` | Names the missing prompt-bound no-bias descriptor argument and dispatch branch at `FeedForward::apply_linear`; no candidate execution or default runtime change |
| No-bias apply-linear callsite descriptor | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-241-apply-linear-callsite-descriptor.json` | Adds a fail-closed optional descriptor argument at `FeedForward::apply_linear` and blocks production execution on prompt/session descriptor construction; no candidate execution or speed claim |
| No-bias prompt/session callsite descriptor | `ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-242-prompt-session-callsite-descriptor.json` | Adds opt-in model/layer propagation for a prompt-bound no-bias descriptor to the exact `FeedForward::apply_linear` callsite, while blocking production warm-session use on descriptor construction and post-decode receipt emission |

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

## Current Performance Profile

This is the current Kaby Lake proof-appliance profile, not a general hardware
claim. The values below are copied from committed receipts and remain scoped to
the recorded i5-8250U host, model artifacts, corpus, backend, and thread
settings.

SLM-CPU-205 records the same profile as a machine-readable consolidation
artifact:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-205-kaby-performance-dashboard-evidence.json
```

That artifact consumes the existing strict Qwen3 Q8_0 appliance receipts,
Qwen2.5 Q8_0 second-model sanity and allocation receipts, the 1/2/4/8-thread
Qwen3 envelope, memory and storage context, and thermal/power fields as
explicitly unavailable. It keeps the default thread recommendation scoped to
recorded evidence: 4 threads had the best total session time and best
steady-decode mean in the committed Qwen3 envelope while generated IDs,
decoded text, strict tokenizer authority, backend identity, and
`fallback_used=false` remained stable. This is not a new timing improvement,
speedup, sustained-throughput, Q4/Q5, server, accelerator, Qwen3.5, or BitNet
QK256 claim.

| Surface | Current Evidence | Interpretation |
| --- | --- | --- |
| Primary appliance model | Qwen3-0.6B Q8_0, SHA `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | Baseline model for Kaby correctness and performance work |
| Second-model sanity | Qwen2.5-0.5B-Instruct Q8_0, SHA `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` | Positive bounded sanity evidence; not a broad model-family claim |
| SmolLM2 | Governed fail-closed by exact metadata-scoped normalization and comparator evidence | Not an answer-ready Kaby model yet |
| Runtime backend | `cpu-rust`, strict GGUF tokenizer metadata, `fallback_used=false` | Required behavior oracle for all optimization PRs |
| Operator thread count | 4 threads | Selected from the bounded thread envelope; fastest recorded total session and steady decode in the committed 1/2/4/8-thread sweep |
| Model/tokenizer load | Model load once: 37,531.292 ms; tokenizer load once: 780.975 ms in the 4-thread operator profile | Cold-load cost is separated from warm prompt timing |
| 4-thread thread-envelope total | 135,679.052 ms total session; 94,416.265 ms warm prompt wall time | Bounded single-run envelope only |
| 4-thread prefill/decode | 64,733.698 ms prefill; 23,725.190 ms decode-total in the thread envelope | Prefill remains the dominant measured phase |
| 4-thread first token | 12,328.333 ms mean; 13,187.0 ms p95 in the thread envelope | First-token latency is high but receipt-backed |
| 4-thread steady decode | 1.963 mean tok/s in the thread envelope | Best recorded steady decode among 1/2/4/8 in this receipt pack; not sustained throughput |
| Operator memory | 3,045,429,248 bytes resident, source `sysinfo_current_process` | Memory context is populated for the operator profile |
| Thermal/power | Fields present; not measured in the thread envelope | No sustained thermal or power claim |
| Storage context | Populated by the operator profile follow-up after Windows verbatim-path handling was fixed | Host free-space context is part of the release evidence surface |

### Thread Envelope

| Threads | Total Session ms | Warm Prompt Wall ms | Prefill ms | Decode Total ms | First Token Mean ms | Steady Decode Mean tok/s | Boundary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 137,179.478 | 94,609.006 | 65,837.089 | 23,650.743 | 12,311.833 | 1.931 | Behavior/determinism passed; no thermal/power/memory sampling |
| 2 | 136,826.761 | 95,562.894 | 65,459.825 | 23,672.087 | 12,405.667 | 1.898 | Behavior/determinism passed; no thermal/power/memory sampling |
| 4 | 135,679.052 | 94,416.265 | 64,733.698 | 23,725.190 | 12,328.333 | 1.963 | Current operator default from bounded evidence |
| 8 | 136,190.928 | 94,979.978 | 65,590.721 | 24,318.391 | 12,332.833 | 1.907 | Behavior/determinism passed; no sustained advantage shown |

The selected default remains 4 threads because it has the best recorded total
session time and steady-decode rate in the committed envelope while preserving
generated IDs across thread counts. The envelope itself explicitly keeps
`sustained_throughput_claim=false`; any future default change needs fresh
receipts with the same model SHA, prompt IDs, generated IDs, decoded text,
backend/kernel identity, tokenizer authority, and `fallback_used=false`.

### Next Safe Optimization Targets

Current receipts point to these bounded next targets:

1. Keep prompt/session buffers pre-sized and reusable, especially prompt prefill
   and decode timing/allocation vectors.
2. Continue reducing `prompt_prefill.forward` and `model.forward` allocation
   churn before broader kernel work.
3. Keep output-head/logits extraction out of the hot path where an exact
   sampler fast path or scratch-buffer path preserves generated IDs.
4. Treat Q8_0 dequant plus GEMV locality as exact-tensor scoped until paired
   Qwen3 and Qwen2.5 receipts prove behavior preservation.
5. Keep source-order or packed-Q8 sidecar runtime paths opt-in until before/after
   receipts prove unchanged generated IDs, decoded text, strict provenance, and
   accepted q_proj numeric evidence.

Q4_K_M and Q4_K_S remain planned expansion targets, not supported runtime
targets. A Q4 artifact is not "supported" until it passes the same gates as the
Qwen3 Q8_0 appliance profile: strict metadata, tokenizer authority,
`fallback_used=false`, constrained corpus, multi-token determinism, warm-session
receipt, operator profile, and bounded timing envelope.

### Allocation Boundary Status

SLM-CPU-160 records the first post-dashboard allocation/buffer-reuse boundary as
blocked rather than changing runtime behavior. The high-level warm-session reuse
fields are already present in committed receipts: session-owned buffers, prompt
token buffers, generated-token buffers, timing buffers, stop-tail buffers, and
bounded KV allocation are receipt-visible. The next lower boundary is
`prompt_prefill.forward` / `model.forward` output storage, where current code
classifies the residual block output-storage path as blocked by Candle-owned
tensor add/output ownership.

The blocker artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/slm-cpu-160-allocation-buffer-reuse-boundary-blocker.json
```

That artifact keeps the Qwen3/Qwen2.5 behavior oracle from SLM-CPU-158, records
that the latest paired behavior receipts have `allocation_audit.enabled=false`,
and names the missing evidence before any runtime allocation change: paired
allocation-audit-enabled Qwen3 and Qwen2.5 before/after receipts, plus a concrete
caller-output-storage shape contract for the exact Candle tensor boundary being
changed.

SLM-CPU-161 captures the allocation-audit-enabled baseline for the same Kaby
CPU route before any new runtime allocation behavior changes. The aggregate
receipts are:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-slm-cpu-161-allocation-audit-baseline.json
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen25-slm-cpu-161-allocation-audit-baseline.json
```

Both receipts use `requested_backend=cpu`, `selected_backend=cpu-rust`,
`runtime_api=cpu`, `fallback_used=false`, strict GGUF tokenizer authority, and
`allocation_audit.enabled=true`. Qwen3 records six prompt receipts from the
existing warm-session corpus; Qwen2.5 records a duplicate-prompt deterministic
baseline from the verified Q8_0 cache artifact. In both cases the measured
dominant allocation hotspot remains `prompt_prefill`, and the next target stays
`residual_block_output_storage_boundary` with status
`layer_output_storage_blocked_by_candle_tensor_add_ops`. This establishes the
baseline required for later before/after allocation PRs; it does not claim
allocation reduction, speedup, sustained throughput, Q4/Q5 support, default
runtime promotion, server/accelerator execution, Qwen3.5 support, or BitNet
QK256 changes.

SLM-CPU-162 turns that blocker into an explicit receipt-visible contract before
changing runtime allocation behavior. New allocation-audit receipts name
`residual_block_output_storage_contract` under
`prompt_prefill_breakdown.forward_boundary`, with the required shape contract
that reusable block output storage must match residual input, branch output,
and block output shape/dtype/device. The contract also records
`TransformerForwardWorkspace` as the future storage owner, keeps
`runtime_allocation_behavior_changed=false`, keeps
`can_fill_layer_output_storage=false`, and preserves the Qwen3/Qwen2.5 strict
CPU behavior gate: model SHA, tokenizer authority, prompt/generated IDs,
decoded text, selected CPU backend/kernel, and `fallback_used=false` must
match before any later allocation-reduction PR can claim improvement.

SLM-CPU-163 consumes that contract and records the first runtime-slice decision:
the residual output storage change remains blocked by the current Candle API,
not by missing Kaby evidence. The blocker artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-163-residual-output-storage-runtime-blocker.json
```

The source audit is specific to `candle-core 0.10.2`: `Tensor::add` returns
`Result<Tensor>`, `broadcast_add` delegates to the owned `add` output path, and
the local Candle source still notes the in-place/pre-allocated variant as a
TODO. SLM-CPU-163 therefore leaves
`runtime_allocation_behavior_changed=false`, requires a future `add_out` /
`broadcast_add_out`-style API or verified backend-local equivalent, and keeps
paired Qwen3/Qwen2.5 strict CPU before/after receipts as the gate before any
allocation-reduction or speed claim.

SLM-CPU-164 moves around that blocked Candle residual-add surface and hardens
the warm-session prompt/session buffer capacity receipt boundary. The receipt
now records per-buffer `needed`, `previous_capacity`, `capacity`,
`capacity_grew`, and `capacity_sufficient` details for prompt tokens,
generated-token vectors, timing vectors, allocation-audit sample vectors,
stop-tail storage, and logits scratch storage. It also records explicit
`capacity_grew_buffers`, `insufficient_buffers`, and
`all_buffers_capacity_sufficient` lists. This makes first-prompt capacity
growth and subsequent prompt reuse machine-checkable without claiming
allocation reduction or speed.

SLM-CPU-165 closed the pre-sizing gate and SLM-CPU-166 implements the narrow
runtime boundary: the warm-session command pre-scans already-rendered/tokenized
prompt metadata and reserves resident prompt/session buffers before the first
prompt loop reset. The aggregate session receipt and each per-prompt
`session_reuse` block record the pre-sizing source, capacity sufficiency, and
that prompt-loop resets reused existing capacity. Paired Qwen3 Q8_0 and Qwen2.5
Q8_0 strict CPU after-change receipts preserve model SHA, tokenizer authority,
prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity,
and `fallback_used=false`. This does not claim allocation reduction, speedup,
or sustained throughput without an explicit before/after receipt comparison.

## Dashboard Refresh State

This refresh is current through the SLM-CPU-168 evidence slice for the
post-SLM-CPU-166 prompt/session buffer pre-sizing receipt comparison gate.

SLM-CPU-167 defines the next performance-lane gate. It does not change runtime
behavior by itself; it requires SLM-CPU-168 or a later slice to commit a Qwen3
Q8_0 and Qwen2.5 Q8_0 strict CPU receipt comparison for the SLM-CPU-166
pre-sizing boundary, or a machine-checkable artifact/cache blocker. That
comparison must preserve model SHA, tokenizer authority, prompt IDs, generated
IDs, decoded text, selected CPU backend/kernel identity, dense hook identity
where applicable, and `fallback_used=false` before any later allocation or
timing claim can use the pre-sizing slice as evidence. The gate must classify
whether pre-sizing changed allocation-audit counters or only made capacity reuse
receipt-visible, then name the next safe allocation hotspot.

SLM-CPU-168 evidence now exists in
`ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-168-presizing-receipt-comparison.json`.
It compares the committed SLM-CPU-161 Qwen3/Qwen2.5 allocation-audit baselines
with post-SLM-CPU-166 warm-session receipts and preserves model SHA, GGUF
tokenizer authority, prompt IDs, generated IDs, decoded text, CPU backend,
prompt-level kernel identity, dense hook selection identity, and
`fallback_used=false`. The comparison classifies prompt/session pre-sizing as
receipt-visible resident capacity reuse: aggregate `prompt_setup` bytes fell,
allocation counts changed, and the dominant hotspot remains
`prompt_prefill.forward`. It makes no allocation-reduction, latency, speedup, or
sustained-throughput claim.

SLM-CPU-169 then makes the `prompt_prefill.forward` frontier receipt-visible
under `slm_cpu_169_frontier`. It keeps runtime allocation behavior unchanged
and names the exact residual-add output-storage blocker: Candle
`Tensor::add` / `Tensor::broadcast_add` return owned `Result<Tensor>` outputs
and expose no caller-provided output-storage parameter for
`transformer.block.output` residual-add reuse.

SLM-CPU-170 consumes that frontier and records the API decision in:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-170-residual-add-storage-api-decision.json
```

The decision is intentionally conservative. The residual-add path has a
concrete integration plan, but no runtime allocation behavior should change
until Candle exposes an `add_out` / `broadcast_add_out`-style API or a verified
backend-local equivalent exists. Any future implementation still needs paired
Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU before/after receipts preserving model
SHA, GGUF tokenizer authority, prompt IDs, generated IDs, decoded text,
selected CPU backend/kernel identity, dense hook identity where applicable, and
`fallback_used=false`. Until that proof exists, the lane should move runtime
allocation work to another measured Qwen3/Qwen2.5 frontier with an available
behavior-preserving API surface. This is not a ripr blocker and no ripr issue is
required.

SLM-CPU-171 selects that next measured frontier rather than guessing. The
selection artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-171-post-residual-allocation-frontier.json
```

The selected frontier is `prompt_tokenize`. It is smaller than
`prompt_prefill.forward`, but unlike the residual-add path it is not blocked by
Candle caller-output-storage support. In the SLM-CPU-168 allocation comparison
it remains large on both accepted appliance models: Qwen3 records
`531,768,688` post-change allocation bytes / `6,386,238` allocations, and
Qwen2.5 records `126,649,361` post-change allocation bytes / `1,520,772`
allocations. SLM-CPU-171 does not change runtime behavior. It sets the next
safe slice to define a prompt-tokenize allocation contract that separates
tokenizer-internal allocations from allocations that can be avoided by exact
rendered-prompt and token-ID reuse, then requires paired Qwen3/Qwen2.5 strict
CPU before/after receipts before any allocation, latency, or timing claim.

SLM-CPU-172 defines that contract in:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-172-prompt-tokenize-allocation-contract.json
```

The contract separates tokenizer-internal allocations, which remain classified
until a tokenizer API exposes caller-owned output storage or cache hooks, from
repo-owned reuse surfaces: rendered prompt text, prompt token IDs, and prompt
token vector capacity. A future runtime slice must key reuse on model SHA,
strict GGUF tokenizer source/authority, template family/source, Qwen no-think
policy, raw and rendered prompt hashes, stop criteria, generation identity, and
prompt ID hash. It must also record cache lookup/hit state and prompt-token
buffer capacities in receipts. SLM-CPU-172 itself keeps
`runtime_allocation_behavior_changed=false`; paired Qwen3/Qwen2.5 strict CPU
before/after receipts are still required before any allocation, latency, or
timing improvement claim.

SLM-CPU-173 makes that contract receipt-visible in warm-session prompt receipts
without changing prompt tokenization behavior:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-173-prompt-tokenize-cache-evidence-fields.json
```

Each prompt receipt now records a `prompt_tokenize_contract` object with the
cache key hash, cache lookup result, rendered prompt hash, prompt IDs hash,
tokenizer-internal allocation classification, repo-owned reuse surfaces, and
prompt-token buffer capacity fields. Aggregate warm-session receipts also record
the cache lookup policy and hit/miss counts. This is evidence plumbing only:
`runtime_allocation_behavior_changed=false`, and paired Qwen3/Qwen2.5 strict CPU
before/after receipts remain required before any allocation, latency, or timing
improvement claim.

SLM-CPU-174 turns that requirement into a concrete paired-receipt gate:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-174-prompt-tokenize-paired-receipt-gate.json
```

The gate requires paired before/after strict CPU warm-session receipts for both
Qwen3-0.6B Q8_0 and Qwen2.5-0.5B-Instruct Q8_0 before any prompt-tokenize
allocation, latency, or timing claim. The comparison must preserve model SHA,
strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text,
selected CPU backend/kernel identity, dense hook identity where applicable, and
`fallback_used=false`. It also requires the SLM-CPU-173
`prompt_tokenize_contract` fields, cache hit/miss evidence, allocation-audit
availability, and prompt-token buffer capacity fields to be present. SLM-CPU-174
is still a gate definition only: it records verified local model SHA observations
and command templates, but it does not change prompt tokenization behavior or
claim allocation, latency, timing, speedup, or sustained throughput improvement.

SLM-CPU-175 consumes that gate with a narrow exact-identity runtime cache
candidate:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-175-prompt-tokenize-exact-identity-cache.json
```

The implementation seeds the same resident warm-session prompt-token cache while
pre-sizing prompt buffers, then reuses the cached token IDs in the prompt loop
when the rendered prompt, BOS policy, and special-token parsing policy are
byte-identical. The paired Qwen3 Q8_0 and Qwen2.5 Q8_0 receipts preserve model
SHA, strict GGUF tokenizer authority, prompt-ID hashes, generated IDs, decoded
text, selected `cpu-rust` backend identity, and `fallback_used=false`. For the
first repeated proof prompt, Qwen3 prompt-tokenize allocation counters drop from
`177264200` bytes / `2128850` allocations to `168` bytes / `1` allocation, and
Qwen2.5 drops from `126653209` bytes / `1520826` allocations to `189` bytes /
`1` allocation. This is scoped to exact prompt-token cache hits in the paired
warm-session receipts; it is not a sustained throughput, decode, prefill,
model-load, server, accelerator, cross-quant, or broad SLM quality claim.

SLM-CPU-176 then classifies the post-prompt-tokenize frontier:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-176-post-prompt-tokenize-frontier.json
```

After SLM-CPU-175, prompt-tokenize is down to one small cache-lookup allocation
in the paired after receipts. The dominant remaining allocation surfaces are
`prompt_prefill.forward` and `model.forward`: Qwen3 records `909929200` bytes /
`918435` allocations in `prompt_prefill.forward` and `90599908` bytes /
`52512` allocations in decode `model.forward`, while Qwen2.5 records
`473847232` bytes / `855631` allocations and `37466028` bytes / `41760`
allocations respectively. The next selected frontier is therefore
`model_forward_owned_tensor_allocation_boundary`. The smaller
`prompt_setup.buffer_reset` surface is deferred because its `40464` bytes /
`334` allocations are receipt/evidence construction while prompt buffers are
already capacity-sufficient; it is not the main runtime SLM limiter.

SLM-CPU-177 records that boundary explicitly:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-177-model-forward-allocation-boundary.json
```

Both Qwen3 and Qwen2.5 after-receipts report
`model_forward_output_storage_api_surface_present_reuse_blocked_by_candle_tensor_ops`.
The current instrumentation names `feed_forward.down_proj.output` as the first
reusable allocation surface, but runtime reuse is still blocked at the
`dense_linear_output_storage_api_boundary`. The downstream owned-output blockers
remain unchanged: final norm is blocked by Candle norm ops that return owned
tensors, residual block output is blocked by Candle tensor add/broadcast-add
ops with no caller-provided output storage API, and full `model.forward` output
reuse remains blocked until those inner surfaces are solved or a fused consumer
avoids returned-tensor materialization. SLM-CPU-177 therefore selects
`dense_linear_output_storage_api_feasibility` for SLM-CPU-178 and keeps runtime
behavior unchanged.

SLM-CPU-178 inspects that feasibility boundary:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-178-dense-linear-output-storage-feasibility.json
```

The read-side state is favorable: `candle_nn::Linear` exposes weight and optional
bias tensors, and existing backend-local Q8 helpers can fill local host output
slices or vectors. The production boundary is still not reusable, however,
because `Linear::forward`, `Tensor::matmul`, optional bias `broadcast_add`, and
the sidecar `Tensor::from_vec` construction all produce or consume owned Candle
Tensor storage. SLM-CPU-178 therefore keeps runtime behavior unchanged and
selects `dense_linear_caller_output_storage_runtime_gate` as the next frontier:
either a real Candle `matmul_out`/`broadcast_add_out`-style API or a fully typed
fused consumer must exist before paired Qwen3/Qwen2.5 receipts can claim even a
bounded allocation improvement.

SLM-CPU-179 turns that frontier into an explicit disabled gate:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-179-dense-linear-caller-output-storage-runtime-gate.json
```

The default runtime remains `existing_candle_linear_forward`; the candidate
selector is not selectable. Enabling the gate requires a real
`matmul_out`/`broadcast_add_out` equivalent, or a typed fused consumer that owns
the downstream dense-linear Tensor consumers, plus paired Qwen3 Q8_0 and
Qwen2.5 Q8_0 strict before/after warm-session receipts. Those receipts must
preserve model SHA, GGUF tokenizer authority, prompt IDs, generated IDs, decoded
text, CPU backend/dense path identity, stop policy, and `fallback_used=false`.
Without allocation counters and timing fields for the affected warm-session
phase, the gate may prove behavior but cannot claim allocation or timing
improvement.

SLM-CPU-180 checks whether the gate can be satisfied by a repo-owned typed fused
dense consumer before waiting on Candle output-storage APIs:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-180-fused-dense-consumer-feasibility.json
```

That route is not narrow enough today. The selected `feed_forward.down_proj.output`
surface is consumed immediately by Candle residual add, trace/workspace recording,
block-output identity, and then next-layer norm or final-norm Tensor consumers. A
local fused consumer would therefore need to own residual-add aliasing and
broadcast semantics, qwen trace identity, workspace boundary identity, next norm
handoff, and strict receipt/checkpoint identity, not just dense-linear output.
SLM-CPU-180 leaves runtime unchanged and moves this branch to a Candle
output-storage dependency or a future multi-stage typed Tensor-view design.

SLM-CPU-181 records that dependency explicitly:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-181-output-storage-dependency-register.json
```

The register keeps the dense-linear caller-output-storage gate disabled and
names the exact blockers before this branch can become runtime work:
`Tensor::matmul_out` or equivalent, `Tensor::broadcast_add_out` / `add_out` or
equivalent, LayerNorm/RMSNorm output-storage support, a typed Tensor handoff
contract for repo-owned alternatives, and receipt-visible output-storage path
identity. It classifies those as dependency evidence, not implemented behavior.
No allocation reduction, timing improvement, speedup, Q4/Q5 support,
accelerator path, Qwen3.5 support, or BitNet QK256 change is claimed.

SLM-CPU-121 records that Qwen3
Q8_0 strict CPU generation now reaches post-guard receipt emission and the
layer-0 `attention.q_proj_output_pre_optional_qnorm` fingerprint after
allocating only the prompt-plus-generation KV capacity required by the tiny
proof run. SLM-CPU-122 then refreshes the shared q_proj-output before/after
comparison: Qwen3 now preserves generated behavior and the f32-le q_proj-output
fingerprint across the before/after pair, but Qwen2.5 preserves generated
behavior while its q_proj-output f32-le fingerprint still changes. That is a
cross-model fail-closed result, not a q_proj sidecar behavior proof, packed-Q8
default-runtime promotion, allocation-performance claim, timing claim,
sustained-throughput claim, or answer-quality claim. SLM-CPU-123 classifies the
remaining Qwen2.5 blocker as unresolved from the existing artifacts because the
before/after traces contain only f32-le SHA256 fingerprints and explicitly set
`contents_dumped=false`. SLM-CPU-124 adds an opt-in bounded q_proj-output dump
surface, captures the full 896-f32 Qwen2.5 before/after vectors for the strict
CPU proof prompt, and classifies the remaining mismatch as small f32 numeric
drift: the first difference is at index 0 with absolute delta `0.000000015`, and
the maximum absolute delta is `0.00008773799999772791` at index 568. Because no
explicit tolerance policy has been accepted for the shared q_proj-output sidecar
proof, SLM-CPU-124 remained a fail-closed diagnostic result rather than a
packed-Q8 behavior proof or performance claim. SLM-CPU-125 then accepts the
minimal next proof gate for this exact boundary only: all Qwen3 values are
fingerprint-identical, all 896 Qwen2.5 values are within an absolute `1e-4`
f32 tolerance, the Qwen2.5 maximum absolute delta is
`0.00008773799999772791`, the mean absolute delta is
`0.000002513893973225678`, the RMS absolute delta is
`0.000006597620119015958`, and both models preserve model SHA, strict tokenizer
authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel
identity, dense hook identity, and `fallback_used=false`. This is an
exact-boundary numeric gate only. It does not promote `packed_q8_sidecar` to the
default runtime and it does not claim allocation reduction, timing improvement,
sustained throughput, Q4/Q5 support, server, accelerator, Qwen3.5, or BitNet
QK256 behavior. SLM-CPU-126 then checks whether that accepted exact-boundary
gate can be used for a fresh allocation or timing experiment. It remains
fail-closed because the local workspace has the verified Qwen3 Q8_0 GGUF, but
the exact Qwen2.5 Q8_0 GGUF required for fresh cross-model before/after receipts
is not present at
`target/slm-cpu-017/cache/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf`.
SLM-CPU-127 restores that exact Qwen2.5 cache artifact in the swarm workspace
without committing the model binary, verifies SHA256
`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`, and
collects fresh Qwen3 and Qwen2.5 before/after strict CPU receipts. The receipt
pack preserves generated IDs, decoded text, strict GGUF tokenizer authority,
`cpu-rust`, `dense-qwen-cpu-reference`, and `fallback_used=false` for both
models. This restores receipt readiness for a later bounded experiment, but it
does not itself claim allocation reduction, timing improvement, speedup,
sustained throughput, default-runtime promotion, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 behavior.
SLM-CPU-128 consumes that restored receipt pack as the first bounded
classification rather than treating availability as improvement. The result is
mixed and fail-closed: Qwen3's one-token `say_four` sample is neutral within a
single-run timing envelope, while Qwen2.5's `math_2_plus_2_brief` sample
regresses on decode-total, first-token, prefill, and tokens-per-second fields.
The receipts do not carry allocation-audit counters, so no allocation reduction
can be evaluated from this pack. The exact-tensor packed-Q8 sidecar remains
opt-in, exact-tensor scoped, and not promoted to the default runtime.
SLM-CPU-129 follows with repeated warm-session allocation-audit receipts for
the same Qwen3 Q8_0 and Qwen2.5 Q8_0 behavior gates. Both models preserve
generated IDs, decoded text, strict GGUF tokenizer authority, `cpu-rust`, and
`fallback_used=false`. The evidence is still fail-closed for optimization:
Qwen3 records allocation counters but remains on `eager_f32_candle` for the
selected path, while Qwen2.5 records opt-in exact-tensor packed-Q8 counter
selection without runtime promotion. The slice therefore proves counter
availability and behavior preservation only; it does not claim allocation
reduction, timing improvement, speedup, sustained throughput, default-runtime
promotion, Q4/Q5, server, accelerator, Qwen3.5, or BitNet QK256 behavior.
SLM-CPU-130 resolves the apparent cross-model selector mismatch as an intended
payload-order guard outcome. The Qwen3 exact q_proj sidecar is payload-bearing
and contract-valid, but its receipt records
`sidecar_payload_order_matches_runtime_shape=false`, so the selector dispatches
and declines all calls while preserving `eager_f32_candle`. Qwen2.5 records
`sidecar_payload_order_matches_runtime_shape=true`, so the same opt-in exact
tensor boundary reaches the packed-Q8 counter path. The next safe Qwen3 work is
a payload-reorder or runtime-shape proof, not a selector relaxation. No
allocation reduction, timing improvement, speedup, sustained throughput,
default-runtime promotion, Q4/Q5, server, accelerator, Qwen3.5, or BitNet QK256
behavior is claimed.
SLM-CPU-131 codifies that next gate as a machine-checkable payload-order proof
surface. For Qwen3 layer-0 `attention.q_proj`, the GGUF source shape is
`[1024, 2048]` while the Candle runtime matrix shape is `[2048, 1024]`; because
the packed Q8_0 bytes are still source-order bytes, runtime selection remains
blocked with `runtime_selection_allowed=false`. The only accepted next step is a
tensor-specific payload reorder/runtime-shape proof, or continuing to preserve
`eager_f32_candle`. This does not claim allocation reduction, timing
improvement, speedup, default-runtime promotion, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-132 resolves that next gate into a more precise fail-closed contract.
For the exact Qwen3 layer-0 `attention.q_proj` tensor, the packed Q8_0 payload
uses 32-value scale blocks in GGUF source row-major order. Consuming that payload
as the transposed runtime matrix shape would regroup values under different
scale blocks, so a pure byte reorder is not an accepted proof. The next safe
runtime path is either an exact-tensor dequantize/requantize equivalence proof
or a source-order Q8_0 kernel; until then the selector keeps Qwen3 on
`eager_f32_candle`. This still makes no allocation, timing, speedup,
default-runtime, Q4/Q5, server, accelerator, Qwen3.5, or BitNet QK256 claim.

SLM-CPU-133 defines the source-order option as the next runtime-disabled
contract. The candidate would consume the Qwen3 q_proj GGUF source-order Q8_0
payload with `source_input_dim=1024`, `source_output_dim=2048`, and a 2048-wide
row-major Q8 block span, accumulating directly into the runtime output vector.
That avoids pure byte-transpose and dequantize/requantize claims, but still
requires an implementation and behavior-equivalence proof before selector use:
accumulator order, block-scale decode, generated IDs, decoded text, selected
backend/kernel identity, dense hook identity, allocation counters, and strict
receipts must match the Qwen3/Qwen2.5 oracle first.

SLM-CPU-134 adds the first runtime-disabled source-order Q8_0 matvec prototype
surface for that contract. The prototype decodes Q8_0 scales/codes inside the
matvec while walking GGUF source-order rows and accumulating into the runtime
output vector; it does not materialize full f32 weights and it does not relax
selector policy. The committed proof is fixture-level and fail-closed for exact
runtime use: Qwen3 and Qwen2.5 before/after strict CPU receipts must still prove
unchanged prompt IDs, generated IDs, decoded text, backend/kernel identity,
dense hook identity, fallback state, and receipt identity before any selector
use, allocation claim, timing claim, or speed claim.

SLM-CPU-135 makes that next gate explicit rather than inferring it from the
prototype artifact. The source-order matvec remains runtime-disabled because the
exact selector-path before/after strict CPU receipt pairs are not yet captured
for both Qwen3 Q8_0 and Qwen2.5 Q8_0. Prior q_proj numeric evidence exists, but
it is not bound to source-order matvec selector receipts. The accepted next step
is therefore receipt capture and comparison for model SHA, GGUF tokenizer
authority, prompt IDs, generated IDs, decoded text, selected backend/kernel
identity, dense hook identity, q_proj numeric evidence, and `fallback_used=false`.
This does not claim allocation reduction, timing improvement, speedup, sustained
throughput, default-runtime promotion, Q4/Q5 support, server, accelerator,
Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-136 attempts to advance that gate and records the precise blocker
instead of using stale baseline/candidate receipts as a proxy. The source-order
matvec prototype is available only as a runtime-disabled proof helper; it is not
wired as a generation-time selector path, it is not bound to the exact runtime
hidden-state q_proj input used by Qwen3/Qwen2.5 strict CPU generation, and
receipts do not yet record `source_order_q8_0_qproj_matvec` as the selected
dense hook candidate with q_proj numeric evidence. The next safe implementation
slice is therefore a default-disabled exact-tensor selector hook plus receipt
identity fields, followed by Qwen3/Qwen2.5 before/after strict CPU receipt pairs.
This still makes no allocation, timing, speedup, sustained-throughput,
default-runtime, Q4/Q5, server, accelerator, Qwen3.5, or BitNet QK256 claim.

SLM-CPU-137 adds that selector identity surface, but SLM-CPU-138 keeps runtime
selector use fail-closed because the proof receipts are not current enough. The
workspace has `models/slm/Qwen3-0.6B-Q8_0.gguf`, but the exact
`models/slm/qwen2.5-0.5b-instruct-q8_0.gguf` cache artifact is missing. The
committed SLM-CPU-127 Qwen3/Qwen2.5 before/after receipts preserve generated
behavior and strict provenance, but they predate the SLM-CPU-137
`source_order_*` identity fields. Fresh cross-model receipts with source-order
candidate path, kernel, dimensions, candidate receipt identity, q_proj numeric
evidence, and `fallback_used=false` are still required before any source-order
selector use. No allocation, timing, speedup, sustained-throughput,
default-runtime, Q4/Q5, server, accelerator, Qwen3.5, or BitNet QK256 claim is
made.

SLM-CPU-139 re-checks that exact-model gate on current main and preserves the
same fail-closed result as a machine-checkable blocker. The workspace still has
the verified Qwen3 Q8_0 GGUF, but still lacks the exact Qwen2.5 Q8_0 model cache
under `models/slm/qwen2.5-0.5b-instruct-q8_0.gguf`; the active slice also
forbids `models/**` and `target/**`, so it cannot restore the binary cache in
this PR. The older SLM-CPU-127 Qwen3/Qwen2.5 receipt pairs remain useful as
behavior-preserving oracle evidence, but they still predate the SLM-CPU-137
source-order identity fields and do not bind q_proj numeric evidence to fresh
current-main strict CPU receipts. The accepted next step is an explicitly scoped
cache restoration or a fresh capture once the exact Qwen2.5 model is available.
No allocation, timing, speedup, sustained-throughput, default-runtime, Q4/Q5,
server, accelerator, Qwen3.5, or BitNet QK256 claim is made.

SLM-CPU-140 converts that blocker into an explicit cache-restoration and
receipt-capture contract. It names the exact Qwen2.5 Q8_0 repo, revision, file,
SHA256, non-committed local cache path, four required Qwen3/Qwen2.5 before/after
receipt outputs, source-order q_proj candidate identity fields, and the
behavior-preservation gate for prompt IDs, generated IDs, decoded text, strict
GGUF tokenizer authority, selected CPU backend/kernel identity, q_proj numeric
evidence, and `fallback_used=false`. It intentionally captures no fresh
receipts, commits no model or target cache artifacts, keeps source-order runtime
selection disabled, and makes no allocation, timing, speedup,
sustained-throughput, default-runtime, Q4/Q5, server, accelerator, Qwen3.5, or
BitNet QK256 claim.

SLM-CPU-141 executes that contract far enough to replace the missing-cache
blocker with a narrower receipt-evidence blocker. The exact Qwen2.5 Q8_0 GGUF
was restored under the non-committed `target/slm-cpu-140/cache/...` path and
verified against SHA256
`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`. Fresh
Qwen3 and Qwen2.5 before/after warm-session receipts preserve strict GGUF
tokenizer authority, `cpu-rust`, `fallback_used=false`, generated token IDs, and
decoded text. The source-order selector gate still remains fail-closed: Qwen3
records the runtime-disabled source-order q_proj candidate identity but lacks
`q_proj_numeric_evidence`, while Qwen2.5 records a payload-bearing q_proj
boundary without the source-order candidate path/kernel/receipt identity and
also lacks `q_proj_numeric_evidence`. No model or target cache artifacts are
committed, and no allocation, timing, speedup, sustained-throughput,
default-runtime, Q4/Q5, server, accelerator, Qwen3.5, or BitNet QK256 claim is
made.

SLM-CPU-142 resolves the absent-field ambiguity from SLM-CPU-141 without
promoting runtime selection. The warm-session `dense_q8_hook` receipt now emits
`source_order_qproj_candidate_identity` for the exact q_proj payload boundary
and `q_proj_numeric_evidence` as an explicit fail-closed object. Qwen3 can
record `candidate_identity_present_runtime_disabled`, while Qwen2.5 can record
`not_source_order_runtime_shape_compatible` instead of looking like a missing
source-order identity bug. The numeric evidence field remains
`not_captured_by_warm_session_receipt`; selector use is still blocked until
fresh Qwen3/Qwen2.5 receipts attach q_proj numeric evidence while preserving
model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded
text, selected CPU backend/kernel identity, and `fallback_used=false`. This is
receipt-surface hardening only. It does not claim allocation reduction, timing
improvement, speedup, sustained-throughput, default-runtime promotion, Q4/Q5,
server, accelerator, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-143 refreshes the exact Qwen3/Qwen2.5 receipt evidence on current main
with the SLM-CPU-142 fields present. The Qwen3 and Qwen2.5 before/after
strict CPU warm-session runs preserve model SHA, strict GGUF tokenizer
authority, prompt IDs, generated IDs, decoded text, selected CPU backend, and
`fallback_used=false`. The result is still fail-closed for selector use:
`source_order_qproj_candidate_identity.status` is
`payload_boundary_present_without_source_order_identity` for both exact models,
and `q_proj_numeric_evidence.status` remains
`not_captured_by_warm_session_receipt`. This is a current-main evidence refresh
and blocker, not an allocation reduction, timing improvement, speedup,
sustained-throughput, default-runtime promotion, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 claim.

SLM-CPU-144 localizes that identity gap. The SLM-CPU-143 after-run commands
enabled `BITNET_DENSE_Q8_RUNTIME_*` for `blk.0.attn_q.weight` but did not also
enable `BITNET_DENSE_Q8_PAYLOAD_*`, so the hook selection receipt contained
`payload_bearing_boundary = null`. The warm-session receipt classifier treated
that JSON null field as an ambiguous payload boundary; SLM-CPU-144 corrects the
classification so a null payload boundary is reported as `no_payload_boundary`.
Selector use remains disabled. A future exact-model receipt capture must set
both the payload and runtime gates, preserve the same strict Qwen3/Qwen2.5
behavior oracle, and still attach q_proj numeric evidence before claiming any
source-order selector use. This is receipt classification and blocker evidence
only, not an allocation reduction, timing improvement, speedup,
sustained-throughput, default-runtime promotion, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 claim.

SLM-CPU-145 performs that exact-model payload/runtime capture with both
`BITNET_DENSE_Q8_PAYLOAD_*` and `BITNET_DENSE_Q8_RUNTIME_*` set for
`blk.0.attn_q.weight`. Qwen3 preserves the strict warm-session behavior oracle
and exposes `layers.0.attention.q_proj.weight:source_order_q8_0_qproj_matvec:runtime_disabled`,
but runtime compute remains declined because the GGUF source payload order does
not match the Candle runtime matrix shape. Qwen2.5 also preserves generated
IDs/text and records 92 opt-in packed q_proj matvec calls for the same exact
tensor. Selector promotion still remains fail-closed because the receipts do
not attach bounded q_proj numeric diff evidence for the runtime path. This is
payload/runtime receipt evidence only, not a default-runtime promotion,
allocation reduction, timing improvement, speedup, sustained-throughput, Q4/Q5,
server, accelerator, Qwen3.5, or BitNet QK256 claim.

SLM-CPU-146 consumes those SLM-CPU-145 receipt oracles and records the current
numeric-evidence gate without treating older diagnostic traces as a proxy for
the payload/runtime warm-session path. The accepted SLM-CPU-125 tolerance policy
still names the right bounded evidence shape, but SLM-CPU-145 warm-session
receipts carry selector identity and behavior preservation only: they do not
attach q_proj-output f32 fingerprints, bounded vector dumps, or before/after
diff fields. The `run` command has q_proj trace dump flags, but
`slm-warm-session` does not yet expose an equivalent trace surface, so the next
safe step is an explicit warm-session q_proj numeric capture surface or artifact
ingest path. Selector promotion remains blocked; no allocation, timing,
speedup, sustained-throughput, default-runtime, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 claim is made.

SLM-CPU-148 consumes the new warm-session q_proj trace surface and attaches the
bounded numeric evidence missing from SLM-CPU-146. Qwen2.5 captures accepted
before/after q_proj-output vector evidence within the SLM-CPU-125 `1e-4`
tolerance while preserving generated IDs/text under the explicit packed-Q8
sidecar gates. Qwen3 preserves generated IDs/text and produces identical
eager-path q_proj-output vectors, but it still records zero selector-selected
and packed-matvec calls because `blk.0.attn_q.weight` is source-order
`[1024, 2048]` while the Candle runtime matrix shape is `[2048, 1024]`.
That is behavior-preserving evidence plus a precise Qwen3 runtime-sidecar
blocker, not a default-runtime promotion, allocation reduction, timing
improvement, speedup, sustained-throughput, Q4/Q5, server, accelerator,
Qwen3.5, or BitNet QK256 claim.

SLM-CPU-149 further narrows the Qwen3 source-order q_proj runtime boundary. The
source-order candidate identity is present and names
`dense-q8-source-order-qproj-matvec`, but current generation receipts still
cannot emit a source-order candidate output vector: the selector declines all
calls, the q_proj trace records only the post-linear eager output boundary, and
there is no receipt-safe binding from the generation-time hidden-state input
slice to the source-order prototype. The next safe slice is therefore a
default-disabled source-order q_proj input/candidate-output capture surface or
selector branch that can compare candidate output against the eager warm-session
oracle before any runtime selection or performance claim.

SLM-CPU-150 adds that default-disabled capture surface for the exact Qwen3
Q8_0 `blk.0.attn_q.weight` / `layers.0.attention.q_proj.weight` layer-0 path.
With `BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_CANDIDATE=1`, the warm-session path
binds the generation-time hidden-state input used by eager q_proj to the
source-order Q8_0 prototype, emits a receipt-safe candidate/eager hash and
bounded-vector diff event, and keeps `eager_f32_candle` as the selected runtime.
The first i5-8250U artifact records candidate length 2048, eager length 2048,
`max_abs_diff_vs_eager=5.735998631`, `mean_abs_diff_vs_eager=0.280192134`,
`rms_abs_diff_vs_eager=0.516290998`, and `fallback_used=false`. This is
diagnostic evidence only: it proves the capture surface and current
source-order/eager mismatch, not allocation reduction, timing improvement,
speedup, sustained throughput, default-runtime promotion, Q4/Q5 support,
server/accelerator execution, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-151 classifies that captured mismatch with a local full-vector trace
without committing the raw trace. The full 2048-value source-order candidate
and eager q_proj oracle differ from index 0, and the sorted-value and
sorted-absolute permutation probes still differ substantially
(`sorted_abs_value_permutation_probe.rms_abs_diff=0.364230255`). The eager
oracle also has a much larger L2 norm and extrema than the candidate
(`21.865050542` vs `7.819040456` L2 norm), so a pure output-index permutation
or hash/diff-construction issue is not sufficient to explain the mismatch. The
next safe boundary is a small source-order q_proj accumulator audit for selected
output indices, including Q8 block index, block scale, quantized byte, input
value, partial sum order, and the corresponding eager oracle slice. This is
still diagnostic evidence only; it does not promote the source-order candidate
to runtime and does not claim allocation reduction, timing improvement, speedup,
sustained throughput, default-runtime promotion, Q4/Q5 support,
server/accelerator execution, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-152 adds the next default-disabled accumulator audit surface for the
same exact Qwen3 Q8_0 q_proj boundary. With
`BITNET_QWEN_TRACE_SOURCE_ORDER_QPROJ_ACCUMULATOR_AUDIT=1`, the trace records
selected output indices with Q8 block index, block scale, quantized byte, input
value, contribution, and partial sum order. The i5-8250U artifact audits output
indices `0`, `1419`, and `1970`; those audited sums recompute the source-order
candidate vector, but the eager Candle q_proj oracle still differs, most sharply
at output index `1419` (`candidate=-0.152995154`,
`eager=-5.88899374`). This narrows the next boundary to source-order payload
interpretation versus eager Candle weight materialization for the matching
row/column slice. It is still diagnostic-only evidence and does not promote the
candidate runtime or claim allocation reduction, timing improvement, speedup,
sustained throughput, Q4/Q5 support, server/accelerator execution, Qwen3.5, or
BitNet QK256 behavior.

SLM-CPU-153 compares those audited source-order q_proj payload terms against
the exact Candle-materialized layer-0 q_proj row slices for the same selected
output indices. The compact artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-slm-cpu-153-candle-qproj-slice-compare.json
```

The compare records:

```text
selected output indices = 0, 1419, 1970
max_abs_diff_candle_vs_eager = 0.000001907
max_abs_diff_source_order_vs_candle = 5.735996723
classification = source_order_payload_to_runtime_row_mapping
```

This shows that direct recomputation from the Candle-materialized q_proj row
slice matches the eager q_proj output within trace tolerance, while the
source-order Q8 payload traversal still diverges. The remaining blocker is the
source-order payload-to-runtime row mapping for `blk.0.attn_q.weight`, not
eager Candle output attribution. It keeps `eager_f32_candle` as the default
runtime and does not claim allocation reduction, timing improvement, speedup,
sustained throughput, default-runtime promotion, Q4/Q5 support,
server/accelerator execution, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-154 proves the selected-row source-order payload-to-runtime row mapping
for the same exact Qwen3 Q8_0 q_proj boundary. The compact artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-slm-cpu-154-source-order-row-mapping-proof.json
```

The proof records:

```text
mapping_rule = runtime_weight_idx=output_index*source_input_dim+input_index
source_order_weight_idx = input_index*source_output_dim+output_index
selected output indices = 0, 1419, 1970
max_abs_diff_mapped_vs_candle = 0.0
max_abs_diff_mapped_vs_eager = 0.000001907
```

This shows that the Q8 payload values reconcile with the Candle row-major
q_proj slices when addressed by the runtime row mapping. The old source-order
candidate accumulator is still diagnostic-only and remains wrong until it uses
that mapping; `eager_f32_candle` remains the default runtime. This does not
claim allocation reduction, timing improvement, speedup, sustained throughput,
default-runtime promotion, Q4/Q5 support, server/accelerator execution,
Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-155 applies that mapping to the runtime-disabled source-order Qwen3
q_proj candidate accumulator. The compact artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-155-source-order-mapped-qproj-candidate.json
```

The trace-gated Qwen3 candidate now records:

```text
source_input_dim = 1024
source_output_dim = 2048
max_abs_diff_vs_eager = 0.000005245
mean_abs_diff_vs_eager = 0.000000113
rms_abs_diff_vs_eager = 0.000000273
selected output indices = 0, 1419, 1970
max_abs_diff_mapped_vs_candle = 0.0
```

The local Qwen3 run preserves strict GGUF tokenizer authority, `cpu-rust`,
`fallback_used=false`, and generated token ID `[17]` while keeping the
source-order candidate runtime-disabled. The Qwen2.5 sanity run preserves the
accepted `math_2_plus_2_brief` behavior, but does not emit the source-order
q_proj candidate stage because that model does not expose the exact Qwen3
source-order runtime-shape boundary. This slice proves only the mapped
trace-gated candidate for the exact Qwen3 q_proj tensor; it does not promote the
candidate to default runtime and does not claim allocation reduction, timing
improvement, speedup, sustained throughput, Q4/Q5 support, server/accelerator
execution, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-156 consumes the SLM-CPU-155 mapped-candidate evidence and records the
next selector-gate decision as fail-closed rather than promoting runtime
selection. The compact artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-156-source-order-qproj-selector-gate.json
```

The selector gate preserves `eager_f32_candle` as the default runtime and keeps
the source-order q_proj candidate runtime-disabled. Runtime promotion remains
blocked until paired Qwen3 and Qwen2.5 strict CPU before/after receipts prove
unchanged model SHA, tokenizer authority, prompt IDs, generated IDs, decoded
text, selected CPU backend/kernel identity, dense hook identity, q_proj numeric
evidence, and `fallback_used=false`. This does not claim allocation reduction,
timing improvement, speedup, sustained throughput, default-runtime promotion,
Q4/Q5 support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-157 captures that paired receipt evidence without changing the runtime
promotion boundary. The compact artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-157-source-order-qproj-receipt-pair.json
```

The Qwen3 before/after pair preserves model SHA, GGUF tokenizer authority,
prompt IDs, generated token ID `[17]`, decoded text `2`, `cpu-rust`,
`dense-f32-candle-linear`, and `fallback_used=false`; its q_proj dump is
fingerprint-identical across 2048 f32 values. The source-order q_proj candidate
identity is receipt-visible, but remains `runtime_disabled`, so
`eager_f32_candle` remains the selected path. The Qwen2.5 before/after pair
preserves model SHA, GGUF tokenizer authority, prompt IDs, generated IDs
`[17, 10, 17, 16819, 220, 19, 13, 151645]`, decoded text
`2+2 equals 4.`, `cpu-rust`, and `fallback_used=false`; its opt-in packed-Q8
sidecar q_proj output remains within the accepted `1e-4` absolute tolerance
with maximum absolute delta `0.00008773799999772791`. This is paired
selector-gate evidence only. It does not promote the source-order q_proj
candidate or packed-Q8 sidecar to default runtime and it does not claim
allocation reduction, timing improvement, speedup, sustained throughput, Q4/Q5
support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior.

SLM-CPU-158 consumes that paired evidence and adds the next explicit opt-in
runtime binding gate for the exact Qwen3 source-order q_proj candidate. The
compact artifact is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-27/qwen3-qwen25-slm-cpu-158-source-order-qproj-runtime-binding-gate.json
```

The default runtime remains `eager_f32_candle`; source-order q_proj execution is
selected only when the exact tensor is explicitly opted in through the dense-Q8
payload/runtime environment gate. With that gate enabled, Qwen3 selects
`source_order_q8_0_qproj_matvec` / `dense-q8-source-order-qproj-matvec` for
`layers.0.attention.q_proj.weight`, preserves generated token ID `[17]` and
decoded text `2`, and keeps `fallback_used=false`. The Qwen3 q_proj numeric
evidence compares 2048 f32 values with maximum absolute delta
`0.000005245000000098088`, within the accepted `1e-4` tolerance. The Qwen2.5
guard pair preserves generated IDs `[17, 10, 17, 16819, 220, 19, 13, 151645]`
and decoded text `2+2 equals 4.`; its compatible packed-Q8 q_proj selector
remains model-specific guard evidence rather than a Qwen3 source-order claim,
with maximum absolute delta `0.00008773799999772791`, also within tolerance.
This slice does not promote any dense-Q8 sidecar path to default runtime and
does not claim allocation reduction, timing improvement, speedup, sustained
throughput, Q4/Q5 support, server/accelerator execution, Qwen3.5, or BitNet
QK256 behavior.

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
attention, output-head math, tokenizer behavior, or generated tokens. It now
feeds a disabled `dense_linear_caller_output_storage_runtime_gate`. That gate is
not selectable by default; it only records the API and receipt evidence required
before future runtime work can attempt caller-owned dense-linear output storage
without weakening the Qwen3/Qwen2.5 strict receipt oracle. It does not claim a
speedup, sustained throughput, Q4/Q5 runtime support, accelerator execution,
Qwen3.5 support, or BitNet QK256 changes.

SLM-CPU-180 closes the local fused-consumer branch for now: the first consumer
after `feed_forward.down_proj.output` is the block residual add, and avoiding the
returned Tensor there would also require owning trace/workspace identity,
block-output Tensor identity, and the next norm/model-forward handoff. That is
larger than a dense-linear output-storage optimization, so this branch remains
disabled until Candle output-storage APIs or a broader typed Tensor-view design
exist.

SLM-CPU-181 turns that into a dependency register rather than another runtime
attempt. The current blockers are exact output-storage APIs for Candle matmul,
residual/broadcast add, and norm outputs, plus a repo-owned typed Tensor handoff
contract if the project chooses not to wait on those APIs. The register also
requires future receipt-visible output-storage path identity and paired strict
Qwen3/Qwen2.5 generated-ID preservation before any allocation or timing claim.

SLM-CPU-182 then checks the narrower no-bias dense-linear frontier:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-182-no-bias-dense-linear-frontier.json
```

That route is not implementation-ready. The code can represent missing-bias
linear roles without materializing zero-bias tensors, and existing strict
Qwen3/Qwen2.5 receipts record zero sidecar bias materialization calls. Those
receipts are not a model-wide bias-presence manifest, though. Attention,
feed-forward, and output-head roles still need exact model-init bias traces or
tensor-manifest evidence before a no-bias fast path can be selected. The
frontier therefore stays runtime-disabled and does not reopen the broader
Candle output-storage blocker from SLM-CPU-181.

SLM-CPU-183 converts that frontier into an explicit manifest gate:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-183-dense-bias-manifest-gate.json
```

The gate requires per-role/layer bias-presence records for attention,
feed-forward, and output-head roles before any future no-bias dense-linear
selector can run. Missing, unknown, or present bias evidence fails closed.
Output-head roles must also record whether the selected path is a dedicated
`lm_head`, tied embeddings, or a transposed head. This remains a manifest
contract only: no runtime behavior changes, no no-bias fast path is selected,
and no allocation, timing, speedup, sustained-throughput, Q4/Q5, accelerator,
Qwen3.5, or BitNet QK256 claim is made.

SLM-CPU-184 checks whether committed evidence can satisfy that gate:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-184-dense-bias-manifest-capture-blocker.json
```

It cannot yet. The code has a `model_init.linear_bias_finish` trace surface and
receipts expose aggregate bias materialization counters, but neither is a
complete per-role/layer manifest for all Qwen3/Qwen2.5 attention, feed-forward,
and output-head roles. Some older sidecar/candidate receipts also show non-zero
bias materialization calls, so the lane cannot infer role-wide bias absence from
zero counters in one strict path. The no-bias branch therefore remains
fail-closed until a real model-init trace export or tensor manifest records
`role_records` with positive bias presence/absence and output-head mode.

SLM-CPU-185 through SLM-CPU-188 narrow that blocker without changing runtime
selection. SLM-CPU-185 adds feed-forward and output-head
`model_init.linear_bias_finish` trace coverage, SLM-CPU-186 records that older
committed traces predate that coverage, and SLM-CPU-187 fixes the remaining
operator prerequisite surface:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-187-post-export-trace-capture-prereq.json
```

The prerequisite artifact names the exact Qwen3 and Qwen2.5 Q8_0 model
identities, strict receipt fields, `BITNET_QWEN_TRACE_JSONL` command shapes, and
expected post-export trace/receipt paths needed before a complete dense-linear
bias `role_records` manifest can be derived. The current worktree has the Qwen3
GGUF under `models/slm`, but the Qwen2.5 Q8_0 artifact is only represented by
prior verification evidence and must be supplied at the exact required path, or
an equivalent operator path, before capture. The no-bias branch remains
fail-closed: no runtime selection change, no allocation/timing claim, no Q4/Q5,
no accelerator/server claim, no Qwen3.5, and no BitNet QK256 work.

SLM-CPU-188 consumes that prerequisite surface and captures the real
post-export trace/receipt bundle:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-188-post-export-bias-trace-capture.json
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-slm-cpu-188-post-export-bias-trace.jsonl
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-slm-cpu-188-post-export-bias-receipt.json
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen25-slm-cpu-188-post-export-bias-trace.jsonl
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen25-slm-cpu-188-post-export-bias-receipt.json
```

The Qwen3 trace records 197 `model_init.linear_bias_finish` events: 28 each
for attention q/k/v/o and feed-forward gate/up/down, plus one tied output-head
event. Every recorded Qwen3 role has `present=false`.

The Qwen2.5 trace records 169 `model_init.linear_bias_finish` events: 24 each
for attention q/k/v/o and feed-forward gate/up/down, plus one direct output-head
event. Qwen2.5 attention q/k/v roles record `present=true`; attention o,
feed-forward roles, and output-head bias record `present=false`. This means the
trace inputs and paired receipts are ready for a bias `role_records` manifest,
but no blanket no-bias dense-linear fast path is selectable.

SLM-CPU-189 derives that manifest:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-189-dense-bias-role-records-manifest.json
role_record_count = 366
qwen3_all_selected_roles_biasless = true
qwen25_attention_qkv_bias_present = true
blanket_no_bias_selector_allowed = false
```

Every record is bound to model SHA, trace file, receipt file, source trace line,
event index, scope, linear role, layer where applicable, bias presence, strict
GGUF tokenizer authority, `cpu-rust`, `runtime_api=cpu`, `fallback_used=false`,
and prompt/generated/text receipt evidence. The manifest is evidence only:
runtime selection remains unchanged and any future selector must be role
specific and fail closed for Qwen2.5 attention q/k/v unless a separate biased
fast path exists.

SLM-CPU-190 turns that manifest into a policy gate, still without enabling a
runtime path:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-190-no-bias-selector-policy-gate.json
runtime_gate_name = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
default_enabled = false
runtime_selection_allowed_in_this_slice = false
eligible_biasless_role_records = 294
blocked_biased_role_records = 72
blanket_no_bias_selector_allowed = false
```

Any later implementation must preserve before/after receipts for model SHA,
strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text,
selected CPU backend/runtime identity, dense path identity, manifest SHA,
role ID, bias presence, and `fallback_used=false`. Missing, duplicate, unknown,
contradictory, or `bias_present=true` role evidence fails closed.

SLM-CPU-191 dry-runs that policy against every role record, still without
selecting a runtime path:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-191-no-bias-selector-dry-run-receipts.json
runtime_gate_name = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
runtime_selection_allowed_in_this_slice = false
role_record_count = 366
eligible_no_bias_candidates = 294
blocked_fail_closed = 72
blocked_because_bias_present_true = 72
```

The dry-run artifact records one decision per role with model SHA, strict
receipt identity, manifest SHA, policy SHA, role ID, bias presence, trace file,
receipt file, and selected decision. It is still evidence only: eligible roles
are future candidates, and blocked roles remain fail-closed. No allocation,
timing, speedup, sustained throughput, Q4/Q5, server, accelerator, Qwen3.5, or
BitNet QK256 behavior is claimed.

SLM-CPU-192 adds the audit-only typed boundary used by later receipts:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-192-no-bias-selector-audit-hook.json
audit_hook_type = DenseLinearNoBiasSelectorAudit
runtime_gate_name = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The hook can report role decisions from exact bias evidence, but it is not a
compute selector. Biasless roles are reported as
`eligible_no_bias_candidate_runtime_disabled`; present or unknown bias fails
closed. The next required proof step is before/after strict warm-session
receipts with the audit hook present and generated IDs/text unchanged.

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

## SLM-CPU-193 No-Bias Selector Preservation Receipts

SLM-CPU-193 binds the SLM-CPU-192 disabled-by-default no-bias selector audit
hook to existing strict Qwen3/Qwen2.5 Q8_0 warm-session receipts:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-193-no-bias-selector-preservation-receipts.json
runtime_gate = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
preservation_verdict.passed = true
```

The receipt overlay preserves the strict identity fields from the paired
warm-session receipts: model SHA, GGUF tokenizer authority, prompt ID hashes,
generated token IDs, decoded text, selected CPU backend/runtime, dense path
identity, and `fallback_used=false`. It also records a representative eligible
Qwen3 no-bias role and a representative fail-closed Qwen2.5 biased role while
keeping blanket no-bias selection blocked.

This is an audit/preservation slice only. It does not rerun model inference,
change compute selection, enable a no-bias fast path, prove allocation
reduction, claim timing improvement or speedup, claim sustained throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256/I2_S paths.

## SLM-CPU-194 No-Bias Fast-Path Implementation Gate

SLM-CPU-194 defines the first safe no-bias dense-linear implementation gate
without adding or selecting a new compute path:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-194-no-bias-fastpath-implementation-gate.json
status = gate_defined_runtime_disabled
runtime_gate = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
runtime_selection_enabled = false
first eligible scope = Qwen3 Q8_0 feed_forward.down_proj layers 0..27
selected role count = 28
```

The first gate is deliberately narrow. It starts with Qwen3
`feed_forward.down_proj` roles because they are no-bias dense-linear roles and
avoid attention head reshaping, q/k/v bias differences, and output-head
token-selection semantics. Qwen2.5 attention q/k/v remains fail-closed because
those roles have `bias_present=true`.

Future runtime use still requires paired strict before/after warm-session
receipts proving unchanged prompt IDs, generated IDs, decoded text, model SHA,
GGUF tokenizer authority, selected CPU backend/runtime, dense path identity,
manifest SHA, role ID, `bias_present`, and `fallback_used=false`.

This slice does not implement a no-bias fast path, change the default runtime,
prove allocation reduction, claim timing improvement or speedup, claim
sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-195 No-Bias Down-Projection Candidate

SLM-CPU-195 adds the first runtime-disabled no-bias dense-linear candidate for
the SLM-CPU-194 scope:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-195-no-bias-down-proj-candidate.json
candidate scope = Qwen3 Q8_0 feed_forward.down_proj layers 0..27
candidate API = bitnet_transformer::dense_linear_no_bias_candidate_forward
candidate kernel = dense-f32-no-bias-matmul-candidate
runtime_gate = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The candidate computes the no-bias linear surface as
`input.matmul(linear.weight().t())` and fails closed if the `Linear` has a bias.
It is not wired into `TransformerModel` execution and does not alter the
selected runtime path. Role selection remains exact and fail-closed: only
Qwen3 Q8_0 `feed_forward.down_proj` roles in layers 0..27 with
`bias_present=false` are eligible candidate records.

Runtime use still requires paired strict before/after warm-session receipts
proving unchanged prompt IDs, generated IDs, decoded text, model SHA, GGUF
tokenizer authority, selected CPU backend/runtime, dense path identity,
manifest SHA, role ID, `bias_present`, and `fallback_used=false`.

This slice does not change default runtime selection, prove allocation
reduction, claim timing improvement or speedup, claim sustained throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256/I2_S paths.

## SLM-CPU-196 No-Bias Candidate Preservation Receipts

SLM-CPU-196 binds the SLM-CPU-195 no-bias candidate API to the existing strict
Qwen3/Qwen2.5 Q8_0 warm-session identity evidence while keeping runtime
selection unchanged:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-196-no-bias-candidate-preservation-receipts.json
candidate artifact = qwen3-qwen25-slm-cpu-195-no-bias-down-proj-candidate.json
candidate API = bitnet_transformer::dense_linear_no_bias_candidate_forward
runtime_gate = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
preservation_verdict.passed = true
```

The overlay binds the candidate to the strict Qwen3 and Qwen2.5 warm-session
receipts used by the no-bias manifest work. It preserves model SHA, GGUF
tokenizer authority, selected CPU backend/runtime, dense path identity,
prompt-generation identity hashes, quality summary status, and
`fallback_used=false`.

This slice does not run a fresh inference pass, enable candidate runtime
selection, prove allocation reduction, claim timing improvement or speedup,
claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-197 No-Bias Runtime-Selection Preflight

SLM-CPU-197 adds a disabled-by-default preflight/audit surface for the
SLM-CPU-195 no-bias down-projection candidate. It can report whether an exact
Qwen3 Q8_0 `feed_forward.down_proj` role would be selectable in a future
receipt-gated experiment, but normal inference still preserves the eager F32
Candle path:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-197-no-bias-runtime-selection-preflight.json
preflight record = bitnet_transformer::DenseLinearNoBiasRuntimeSelectionPreflight
runtime_gate = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate path = qwen3_feed_forward_down_proj_no_bias_candidate
candidate kernel = dense-f32-no-bias-matmul-candidate
```

The preflight fails closed when the gate is not requested, when paired strict
before/after warm-session receipts are missing, or when the role is outside the
exact Qwen3 Q8_0 down-projection scope. If the gate is requested and receipts
are present, the result is still only
`would_select_candidate_in_receipt_gated_experiment`; normal inference remains
unmodified.

This slice does not run fresh inference, enable candidate runtime selection in
normal inference, prove allocation reduction, claim timing improvement or
speedup, claim sustained throughput, broaden Q4/Q5 support, or touch server,
GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-198 No-Bias Runtime Experiment Gate

SLM-CPU-198 consumes the SLM-CPU-197 preflight and records that the first
receipt-gated runtime experiment remains blocked until the runtime descriptor
and receipts can carry exact no-bias role identity:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-198-no-bias-runtime-experiment-gate.json
runtime_gate = BITNET_DENSE_NO_BIAS_LINEAR_ENABLE
runtime_gate_default_enabled = false
experiment_verdict.status = blocked
experiment_verdict.reason = runtime_descriptor_and_before_after_receipts_missing
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate path = qwen3_feed_forward_down_proj_no_bias_candidate
candidate kernel = dense-f32-no-bias-matmul-candidate
```

The current dense runtime hook descriptor is still sidecar-Q8 oriented and does
not carry the manifest-bound no-bias fields required for a safe runtime
experiment: model SHA, manifest SHA, role ID, layer, scope, linear, and
`bias_present=false`. The feed-forward callsite can derive the dense tensor
name, but it does not yet receive a no-bias runtime descriptor that proves the
exact Qwen3 Q8_0 `feed_forward.down_proj` role before compute selection.

Runtime selection therefore remains blocked until a future slice adds that
descriptor and captures fresh paired strict warm-session receipts proving
unchanged prompt IDs, generated IDs, decoded text, model SHA, GGUF tokenizer
authority, selected CPU backend/runtime, dense path identity, role identity,
manifest SHA, and `fallback_used=false`.

This slice does not wire the no-bias candidate into normal inference, change
the default runtime, prove allocation reduction, claim timing improvement or
speedup, claim sustained throughput, broaden Q4/Q5 support, or touch server,
GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-199 No-Bias Runtime Descriptor Contract

SLM-CPU-199 defines the manifest-bound descriptor contract that SLM-CPU-198
identified as missing before any no-bias runtime experiment can be safely
attempted:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-199-no-bias-runtime-descriptor-contract.json
contract record = bitnet_transformer::DenseLinearNoBiasRuntimeDescriptorContract
source preflight = bitnet_transformer::DenseLinearNoBiasRuntimeSelectionPreflight
ready decision = descriptor_contract_ready_runtime_disabled
blocked decision = blocked_fail_closed
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate path = qwen3_feed_forward_down_proj_no_bias_candidate
candidate kernel = dense-f32-no-bias-matmul-candidate
```

The contract carries the exact identity that a future `FeedForward::apply_linear`
gate must preserve in receipts: model SHA, quant format, manifest SHA, role ID,
layer, scope, linear, `bias_present=false`, candidate path/kernel, selected
path/kernel, runtime gate state, and `fallback_used=false`. It fails closed when
descriptor fields are missing, receipt identity fields are missing, quant format
is not Q8_0, bias evidence is unknown or true, or the source preflight is not
receipt-gate selectable.

This slice still does not execute the no-bias candidate, change default runtime
selection, prove generated ID/text preservation for a runtime experiment, prove
allocation reduction, claim timing improvement or speedup, claim sustained
throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620,
Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-200 No-Bias Apply-Linear Audit Boundary

SLM-CPU-200 consumes the SLM-CPU-199 descriptor contract and adds the
apply-linear audit boundary needed before any future FeedForward runtime
selector can be considered:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-200-no-bias-apply-linear-audit-boundary.json
audit record = bitnet_transformer::DenseLinearNoBiasApplyLinearAuditBoundary
source descriptor = bitnet_transformer::DenseLinearNoBiasRuntimeDescriptorContract
observed decision = descriptor_observed_at_apply_linear_runtime_disabled
blocked decision = blocked_fail_closed
callsite tensor = layers.3.feed_forward.down_proj.weight
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The audit boundary proves that an exact Qwen3 Q8_0 `feed_forward.down_proj`
descriptor can be matched to the dense tensor callsite while still preserving
the eager F32 Candle runtime. It fails closed if the tensor callsite does not
match the descriptor role, if the runtime gate is not requested, or if the
source descriptor already records fail-closed conditions.

Runtime selection remains blocked on descriptor/receipt emission wiring and
fresh paired strict warm-session receipts proving unchanged prompt IDs,
generated IDs, decoded text, model SHA, GGUF tokenizer authority, selected CPU
backend/runtime, dense path identity, role ID, layer, scope, linear,
`bias_present=false`, selected path/kernel, and `fallback_used=false`.

This slice does not execute the no-bias candidate, change default runtime
selection, prove allocation reduction, claim timing improvement or speedup,
claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-201 No-Bias Apply-Linear Receipt Boundary

SLM-CPU-201 consumes the SLM-CPU-200 apply-linear audit boundary and defines
the receipt-emission surface required before any future no-bias runtime
experiment can select the Qwen3 Q8_0 `feed_forward.down_proj` candidate:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-201-no-bias-apply-linear-receipt-boundary.json
receipt record = bitnet_transformer::DenseLinearNoBiasApplyLinearReceiptBoundary
source audit record = bitnet_transformer::DenseLinearNoBiasApplyLinearAuditBoundary
ready decision = receipt_emission_boundary_ready_runtime_disabled
blocked decision = blocked_fail_closed
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
```

The required receipt identity now names the model SHA, quant format, manifest
SHA, role ID, layer, scope, linear role, `bias_present=false`, callsite tensor,
selected eager path/kernel, candidate path/kernel, runtime gate state,
`runtime_api=cpu`, `selected_backend=cpu-rust`, `fallback=false`, prompt ID
digest, generated ID digest, and decoded text digest. The boundary fails closed
if the apply-linear descriptor was not observed, if CPU/backend/fallback
identity is wrong, or if any prompt/generated/text receipt identity is missing.

Runtime selection remains blocked on wiring this boundary into strict
warm-session receipt emission and capturing fresh before/after receipts proving
unchanged prompt IDs, generated IDs, decoded text, model SHA, GGUF tokenizer
authority, backend/runtime identity, dense path identity, role ID, layer,
scope, linear, selected path/kernel, candidate path/kernel, runtime gate state,
and `fallback_used=false`.

This slice does not execute the no-bias candidate, change default runtime
selection, prove allocation reduction, claim timing improvement or speedup,
claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-202 No-Bias Apply-Linear Before/After Receipt Gate

SLM-CPU-202 consumes the SLM-CPU-201 receipt boundary and adds the next
before/after strict warm-session receipt comparison gate, still with candidate
execution disabled:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-202-no-bias-apply-linear-before-after-receipts.json
receipt gate record = bitnet_transformer::DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate
input record = bitnet_transformer::DenseLinearNoBiasApplyLinearReceiptBoundary
ready decision = before_after_receipt_gate_ready_runtime_disabled
blocked decision = blocked_fail_closed
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
```

The gate compares two receipt-boundary records and fails closed unless
descriptor/callsite identity, model SHA, quant format, manifest SHA, role ID,
layer, scope, linear role, `bias_present=false`, selected eager path/kernel,
candidate path/kernel, `runtime_api=cpu`, `selected_backend=cpu-rust`,
`fallback=false`, prompt ID digest, generated ID digest, and decoded-text digest
are all present and preserved across the before/after pair.

This slice defines the comparison gate rather than capturing fresh runtime
receipts. Runtime selection remains blocked on real Qwen3 Q8_0 and Qwen2.5 Q8_0
strict warm-session receipt pairs that carry the gate fields through the
receipt emitter. Candidate execution remains disabled until a later explicit
runtime-selection PR supplies those receipts.

This slice does not execute the no-bias candidate, change default runtime
selection, prove allocation reduction, claim timing improvement or speedup,
claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-203 No-Bias Apply-Linear Receipt Emitter Gate

SLM-CPU-203 consumes the SLM-CPU-202 before/after receipt comparison gate and
wires the disabled no-bias apply-linear gate surface into strict warm-session
receipts:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-28/qwen3-qwen25-slm-cpu-203-no-bias-apply-linear-receipt-emitter-gate.json
receipt field = dense_no_bias_apply_linear_gate
record type = bitnet_transformer::DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate
decision = blocked_receipt_emitter_gate_defined_runtime_disabled
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_path = qwen3_feed_forward_down_proj_no_bias_candidate
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
runtime_api = cpu
selected_backend = cpu-rust
fallback_used = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
```

The warm-session receipt emitter can now carry descriptor/callsite identity,
before/after receipt-pair presence, model SHA, quant format, manifest SHA,
role ID, tensor name, bias state, selected eager path/kernel, candidate
path/kernel, runtime gate state, `runtime_api=cpu`, `selected_backend=cpu-rust`,
`fallback=false`, prompt ID digest, generated ID digest, and decoded-text digest
for the disabled no-bias apply-linear boundary.

This slice records the exact remaining blocker instead of pretending fresh
runtime evidence exists:

```text
fresh_qwen3_qwen25_before_after_warm_session_receipts_with_no_bias_gate_fields_missing
```

Candidate execution remains disabled. Runtime selection remains `eager_f32_candle`
and `dense-f32-candle-linear`.

This slice does not execute the no-bias candidate, change default runtime
selection, prove allocation reduction, claim timing improvement or speedup,
claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-204 No-Bias Apply-Linear Runtime Receipt Pairs

SLM-CPU-204 consumes the SLM-CPU-203 warm-session receipt-emitter gate and
attempts to advance from an emitter surface to real before/after receipt-pair
evidence. The current worktree cannot satisfy that gate because the Qwen2.5
Q8_0 GGUF required for the paired-model proof is not present:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-204-no-bias-apply-linear-runtime-receipt-pairs.json
decision = blocked_missing_required_model_artifact
qwen3_q8_model_present = true
qwen3_q8_sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
qwen25_q8_model_present = false
missing = models/slm/Qwen2.5-0.5B-Instruct-Q8_0.gguf
receipt_pair_capture.captured = false
receipt_pair_capture.capture_attempted = false
```

The gate requires fresh Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict
warm-session receipt pairs carrying the disabled no-bias apply-linear fields.
Partial Qwen3-only evidence would not prove the paired behavior oracle, so this
slice records the exact missing input instead of claiming preservation.

The normal runtime remains unchanged:

```text
runtime_api = cpu
selected_backend = cpu-rust
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_path = qwen3_feed_forward_down_proj_no_bias_candidate
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
fallback_used = false
```

This slice does not execute the no-bias candidate, change default runtime
selection, prove generated-ID preservation for a runtime experiment, claim
allocation reduction, claim timing improvement or speedup, claim sustained
throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO,
UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-206 No-Bias Apply-Linear Prerequisite Refresh

SLM-CPU-206 refreshes the SLM-CPU-204 model-artifact prerequisite after the
Kaby performance dashboard consolidation:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-206-no-bias-apply-linear-prereq-refresh.json
decision = verified_cache_artifact_available_canonical_qwen25_path_missing_receipts_not_captured
qwen3_canonical_artifact_verified = true
qwen3_sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
qwen25_exact_cache_artifact_verified = true
qwen25_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
qwen25_canonical_models_slm_path_present = false
fresh_before_after_receipts_present = false
```

The important change from SLM-CPU-204 is narrower evidence: the exact Qwen2.5
Q8_0 artifact is available in local uncommitted `target` caches and matches
the accepted SHA, but it is still absent from the canonical `models/slm` path.
No model binary is committed. Fresh before/after strict warm-session receipt
pairs remain a later slice and must use explicit model paths while preserving
model SHA, tokenizer authority, prompt/generated IDs, decoded text, CPU
backend identity, `selected_path=eager_f32_candle`, candidate/gate identity,
and `fallback_used=false`.

This slice does not execute the no-bias candidate, change default runtime
selection, prove allocation reduction, claim timing improvement or speedup,
claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-207 No-Bias Apply-Linear Receipt Capture Blocker

SLM-CPU-207 verifies that the explicit Qwen3 and Qwen2.5 Q8_0 model paths are
available for receipt capture, but blocks the actual before/after run because
the current warm-session aggregate cannot bind a real no-bias apply-linear gate
object:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-207-no-bias-apply-linear-receipt-capture-blocker.json
decision = blocked_emitter_passes_none_for_no_bias_apply_linear_gate
qwen3_sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
qwen25_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
warm_session_callsite = crates/bitnet-cli/src/main.rs:9659
callsite = slm_warm_session_no_bias_apply_linear_receipt_emitter_gate(None)
fresh_before_after_receipts_captured = false
candidate_execution_enabled = false
default_runtime_changed = false
```

This is a source-level blocker, not a model-artifact blocker. Running the long
Qwen3 and Qwen2.5 warm-session pairs now would still emit the disabled
placeholder gate, leaving model-bound prompt/generated/text digests,
descriptor/callsite identity, and role/tensor fields unpopulated. The next safe
slice is to wire a real `DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate` or
equivalent receipt-safe boundary into the warm-session aggregate, then rerun
explicit-path before/after receipts while keeping candidate execution disabled.

This slice does not execute the no-bias candidate, change default runtime
selection, prove generated-ID preservation for a no-bias experiment, claim
allocation reduction, claim timing improvement or speedup, claim sustained
throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO,
UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-208 No-Bias Apply-Linear Gate Wiring

SLM-CPU-208 consumes the SLM-CPU-207 source-level blocker by constructing a real
`DenseLinearNoBiasApplyLinearBeforeAfterReceiptGate` object for eligible
Qwen3/Qwen2.5 Q8_0 `cpu-rust` warm-session aggregates and passing it to the
aggregate no-bias apply-linear receipt emitter:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-208-no-bias-apply-linear-gate-wiring.json
decision = gate_object_wired_runtime_disabled_before_after_receipts_missing
eligible_quant = Q8_0
eligible_architectures = qwen2, qwen3
runtime_api = cpu
selected_backend = cpu-rust
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_execution_enabled = false
default_runtime_changed = false
```

The gate now binds model SHA, quant format, manifest digest, role/tensor
identity, prompt/generated/text digests, backend identity, selected path/kernel,
candidate path/kernel, runtime-gate state, and `fallback_used=false`. It remains
fail-closed because fresh before/after strict warm-session receipt pairs have
not yet been captured through this wired boundary:

```text
before_after_receipts_present = false
decision = blocked_pending_before_after_warm_session_receipts
remaining_runtime_selection_blocker = fresh_qwen3_qwen25_before_after_warm_session_receipts
fail_closed_conditions = before_after_receipts_missing
```

The next safe slice is to rerun explicit-path Qwen3 Q8_0 and Qwen2.5 Q8_0
before/after strict warm-session receipts through this wired gate object while
candidate execution remains disabled. This slice does not execute the no-bias
candidate, change default runtime selection, prove generated-ID preservation for
a no-bias experiment, claim allocation reduction, claim timing improvement or
speedup, claim sustained throughput, broaden Q4/Q5 support, or touch server,
GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-209 No-Bias Apply-Linear Receipt Pairs

SLM-CPU-209 consumes the SLM-CPU-208 wired gate by capturing fresh explicit-path
Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict warm-session receipt pairs on
the i5-8250U:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-209-no-bias-apply-linear-receipt-pairs.json
qwen3_before = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-slm-cpu-209-no-bias-before.json
qwen3_after = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-slm-cpu-209-no-bias-after.json
qwen25_before = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen25-slm-cpu-209-no-bias-before.json
qwen25_after = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen25-slm-cpu-209-no-bias-after.json
runtime_api = cpu
selected_backend = cpu-rust
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
runtime_gate_requested_enabled = false
fallback_used = false
```

For both model families, the before/after receipts preserve model SHA, quant
format, strict GGUF tokenizer authority, prompt IDs digest, generated IDs
digest, decoded-text digest, selected backend/path/kernel identity, candidate
path/kernel identity, descriptor/callsite identity, and `fallback_used=false`.
Qwen3 emits token `17` / `2` in both receipts; Qwen2.5 emits token `19` / `4`
in both receipts.

Each individual warm-session receipt still records
`before_after_receipts_present=false` because candidate execution and normal
runtime selection remain disabled. The SLM-CPU-209 pair artifact supplies the
explicit before/after evidence without enabling the candidate path, changing the
default runtime, claiming allocation reduction, claiming timing improvement or
speedup, claiming sustained throughput, broadening Q4/Q5 support, or touching
server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-210 No-Bias Apply-Linear Runtime Experiment Blocker

SLM-CPU-210 consumes the SLM-CPU-209 receipt pairs and records that the first
explicit-gate no-bias apply-linear runtime experiment is still blocked at the
runtime selector boundary:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-210-no-bias-apply-linear-runtime-experiment-blocker.json
decision = blocked_missing_runtime_selector_identity
callsite = bitnet_transformer::FeedForward::apply_linear
descriptor = bitnet_transformer::DenseLinearRuntimeHookDescriptor
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
fallback_used = false
```

The existing transformer callsite receives the input tensor, `Linear`,
projection name, raw tensors, and dense-linear hook registry. That is enough to
observe the tensor callsite and preserve the eager F32 Candle path, but not
enough to safely select a no-bias candidate under the SLM-CPU-209 proof
contract. The selector still lacks receipt-bound identity for model SHA,
architecture, quant format, strict tokenizer authority, runtime API, selected
backend, fallback status, prompt/generated/text digests, manifest digest, and a
per-callsite candidate execution receipt.

The next safe runtime slice is therefore a receipt-bound selector descriptor
that carries those identity fields into `FeedForward::apply_linear`, followed by
fresh candidate-off/candidate-on strict warm-session receipt pairs. This slice
does not execute the no-bias candidate, change the default runtime, claim
generated-ID preservation for a candidate-on experiment, claim allocation
reduction, claim timing improvement or speedup, claim sustained throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or
BitNet QK256/I2_S paths.

## SLM-CPU-211 Receipt-Bound No-Bias Selector Descriptor

SLM-CPU-211 consumes the SLM-CPU-210 selector blocker by adding a fail-closed
receipt-bound selector descriptor surface:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-211-receipt-bound-no-bias-selector-descriptor.json
descriptor = bitnet_transformer::DenseLinearNoBiasReceiptBoundSelectorDescriptor
carrier = bitnet_transformer::DenseLinearRuntimeHookDescriptor.receipt_bound_no_bias_selector
callsite = bitnet_transformer::FeedForward::apply_linear
decision = receipt_bound_selector_descriptor_added_runtime_disabled
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The descriptor can carry the SLM-CPU-209 model SHA, architecture, quant format,
strict GGUF tokenizer authority, runtime API, selected backend, fallback status,
prompt/generated/text digests, manifest digest, before/after receipt-pair
identity, candidate path/kernel, and exact `feed_forward.down_proj` callsite
identity for Qwen3 Q8_0 and Qwen2.5 Q8_0. It remains runtime-disabled and
fail-closed if any receipt identity is missing, if Qwen2.5 candidate policy is
absent, if fallback is used, or if selected path/kernel drift away from the
eager F32 Candle baseline.

This is not a runtime-selection PR. It does not execute the no-bias candidate,
change default inference, claim generated-ID preservation for candidate-on
execution, claim allocation or timing improvement, claim speedup or sustained
throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620,
Qwen3.5, or BitNet QK256/I2_S paths. The next runtime gate still requires fresh
candidate-off/candidate-on strict warm-session receipts.

## SLM-CPU-212 No-Bias Candidate-On Receipt Blocker

SLM-CPU-212 consumes the SLM-CPU-211 descriptor surface and records that the
first candidate-off/candidate-on strict warm-session receipt pair is still
blocked at session-to-runtime propagation:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-212-no-bias-candidate-on-off-receipt-blocker.json
decision = blocked_missing_receipt_bound_selector_propagation_to_runtime_hooks
hook_field = DenseLinearRuntimeHookDescriptor.receipt_bound_no_bias_selector
hook_construction = bitnet_models::bitnet::dense_q8_runtime_hooks_from_sidecars
warm_session_gate = bitnet_cli::slm_warm_session_no_bias_apply_linear_gate_for_session
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The warm-session gate has the model/tokenizer/backend/fallback and
prompt/generated/text digest identity after prompt execution, but the model
dense-linear hook registry is built from sidecar descriptors before that
identity exists and currently carries `receipt_bound_no_bias_selector = null`.
Therefore `FeedForward::apply_linear` still cannot prove the SLM-CPU-209
receipt identity at candidate execution time.

The next safe slice is a session-scoped selector propagation boundary or a
per-callsite candidate execution receipt emitter. It must remain fail-closed
until the explicit runtime gate is requested and candidate-off/candidate-on
strict warm-session receipts prove generated IDs and decoded text are preserved.
This blocker makes no allocation, timing, speedup, sustained throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claim.

## SLM-CPU-213 Selector Propagation Boundary

SLM-CPU-213 consumes the SLM-CPU-212 blocker and defines the fail-closed
boundary between warm-session receipt identity and the `FeedForward::apply_linear`
runtime selector:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-213-selector-propagation-boundary.json
decision = blocked_fail_closed
reason = receipt_bound_selector_identity_cannot_reach_apply_linear_before_candidate_execution
remaining_blocker = session_hook_registry_mutation_point_or_per_callsite_candidate_receipt_emitter
prompt_digest_lifetime = available_after_warm_session_prompt_execution
hook_registry_owner = bitnet_models::bitnet::dense_q8_runtime_hooks_from_sidecars
apply_linear_callsite = bitnet_transformer::FeedForward::apply_linear
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The SLM-CPU-211 descriptor can represent exact Qwen3/Qwen2.5 Q8_0 model,
strict GGUF tokenizer, CPU backend, fallback, prompt/generated/text digest,
selected-path, and candidate-path identity. The missing piece is still the
safe ownership point: production hook construction happens before warm-session
prompt digests exist, and the current callsite has neither a session-scoped
hook mutation point nor a per-callsite candidate execution receipt emitter.

Candidate execution therefore remains disabled. This slice does not change the
default runtime, claim generated-ID preservation for candidate-on execution,
claim allocation or timing improvement, claim speedup or sustained throughput,
broaden Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620,
Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-214 No-Bias Selector Attachment Point

SLM-CPU-214 consumes the selector propagation boundary and selects the safer
attachment strategy: a per-callsite candidate receipt emitter, not a
session-scoped mutation of the shared model hook registry.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-214-no-bias-selector-attachment-point.json
decision = per_callsite_candidate_receipt_emitter_defined_fail_closed
reason = descriptor_identity_reaches_apply_linear_but_candidate_on_proof_is_incomplete
remaining_blocker = explicit_gate_and_candidate_off_on_generated_id_preservation_receipts
callsite = bitnet_transformer::FeedForward::apply_linear
boundary = bitnet_transformer::DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The per-callsite boundary binds the receipt-bound selector descriptor to the
exact dense tensor name and callsite identity while carrying model SHA,
architecture, Q8_0 quant format, strict GGUF tokenizer authority,
`runtime_api=cpu`, `selected_backend=cpu-rust`, `fallback_used=false`,
prompt/generated/text digests, selected path/kernel, and candidate path/kernel.
This avoids prompt-bound identity leaking through a shared session hook registry.

Candidate execution still requires an explicit runtime gate plus candidate-off
and candidate-on strict warm-session receipts proving prompt IDs, generated IDs,
and decoded text are preserved. This slice makes no allocation, timing, speedup,
sustained throughput, Q4/Q5, server/accelerator, Qwen3.5, or BitNet QK256 claim.

## SLM-CPU-215 No-Bias Candidate Off/On Receipt Gate

SLM-CPU-215 consumes the per-callsite selector attachment point and defines the
receipt-pair gate that must pass before any no-bias apply-linear candidate
execution can be attempted.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-215-no-bias-candidate-off-on-receipt-gate.json
decision = candidate_off_on_receipt_pair_gate_defined_fail_closed
reason = candidate_off_on_receipt_pair_incomplete
remaining_blocker = candidate_on_strict_warm_session_receipt_artifact
boundary = bitnet_transformer::DenseLinearNoBiasCandidateOffOnReceiptPairGate
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

The gate is scoped only to Qwen3 Q8_0 and Qwen2.5 Q8_0
`feed_forward.down_proj`. Existing candidate-off evidence carries real GGUF
model SHA, strict GGUF tokenizer authority, `runtime_api=cpu`,
`selected_backend=cpu-rust`, `fallback_used=false`, selected path/kernel,
candidate path/kernel, prompt IDs, generated IDs, decoded text, and exact
per-callsite identity through the SLM-CPU-214 boundary. The missing evidence is
a candidate-on strict warm-session receipt preserving the same IDs and decoded
text under the explicit gate.

This slice does not execute the no-bias candidate, change default runtime,
claim generated-ID preservation for candidate-on execution, claim allocation or
timing improvement, claim speedup or sustained throughput, broaden Q4/Q5 support,
or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths.

## SLM-CPU-216 No-Bias Candidate-On Behavior Evidence Gate

SLM-CPU-216 consumes the SLM-CPU-215 receipt-pair gate and records the next
candidate-on behavior boundary. The first candidate-on attempt still fails
closed before runtime selection because `FeedForward::apply_linear` has no
candidate-on attachment point that can emit the strict warm-session receipt
fields required by the lane.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-216-no-bias-candidate-on-behavior-evidence-gate.json
decision = candidate_on_behavior_evidence_gate_defined_fail_closed
reason = candidate_on_behavior_evidence_or_runtime_attachment_incomplete
remaining_blocker = candidate_on_apply_linear_runtime_attachment_point
boundary = bitnet_transformer::DenseLinearNoBiasCandidateOnBehaviorEvidenceGate
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

The boundary keeps the exact SLM-CPU-216 scope: Qwen3 Q8_0 and Qwen2.5 Q8_0
`feed_forward.down_proj`, strict GGUF tokenizer authority, `runtime_api=cpu`,
`selected_backend=cpu-rust`, `fallback_used=false`, and receipt-visible selected
and candidate path/kernel identity. It does not turn on the candidate path. It
names the missing attachment point and receipt fields required before a later
slice may capture candidate-off/candidate-on behavior evidence.

This slice does not execute the no-bias candidate, change default runtime,
claim generated-ID preservation for candidate-on execution, claim allocation or
timing improvement, claim speedup or sustained throughput, broaden Q4/Q5 support,
or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths.

## SLM-CPU-217 No-Bias Candidate Runtime Attachment Boundary

SLM-CPU-217 consumes the SLM-CPU-216 behavior gate and defines the explicit
candidate-on runtime attachment boundary for the same Qwen3/Qwen2.5 Q8_0
`feed_forward.down_proj` scope. It records the remaining blocker as runtime
ownership: `FeedForward::apply_linear` still does not have a safe candidate
runtime owner that can call the no-bias candidate and emit strict candidate-on
receipt fields for the same per-callsite descriptor identity.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-217-no-bias-candidate-runtime-attachment.json
decision = candidate_runtime_attachment_defined_fail_closed
reason = apply_linear_runtime_ownership_or_receipt_emission_incomplete
remaining_blocker = candidate_runtime_owner
boundary = bitnet_transformer::DenseLinearNoBiasCandidateRuntimeAttachmentBoundary
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

The boundary is still fail-closed. It makes the required runtime owner and
candidate-on receipt emitter explicit, but it does not execute the no-bias
candidate, does not change default runtime, does not claim generated-ID
preservation for candidate-on execution, and does not make allocation, timing,
speedup, sustained-throughput, Q4/Q5, server/accelerator, Qwen3.5, or BitNet
QK256 claims.

## SLM-CPU-218 No-Bias Candidate Runtime Owner Boundary

SLM-CPU-218 consumes the SLM-CPU-217 attachment boundary and records
`FeedForward::apply_linear` as the fail-closed owner boundary for a future
explicitly gated no-bias candidate call. The owner boundary has the callsite
inputs, `Linear` weight access, and callable no-bias candidate surface needed
for Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj`, but it still lacks a
same-callsite candidate-on receipt emitter and fresh strict candidate-off/on
receipts.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-218-no-bias-runtime-owner-boundary.json
decision = candidate_runtime_owner_defined_fail_closed
reason = same_callsite_candidate_on_receipt_emission_incomplete
remaining_blocker = same_callsite_candidate_on_receipt_emitter
boundary = bitnet_transformer::DenseLinearNoBiasCandidateRuntimeOwnerBoundary
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

The boundary does not execute the no-bias candidate, does not change default
runtime, does not claim generated-ID preservation for candidate-on execution,
and does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. SLM-CPU-219 can safely
consume this boundary to wire or block the same-callsite receipt emitter.

## SLM-CPU-219 No-Bias Same-Callsite Receipt Emitter Boundary

SLM-CPU-219 consumes the SLM-CPU-218 runtime owner boundary and records the
same-callsite receipt-emitter boundary for the Qwen3/Qwen2.5 Q8_0
`feed_forward.down_proj` no-bias candidate. The boundary is now local to
`FeedForward::apply_linear`, but it remains fail-closed because fresh strict
candidate-off and candidate-on receipts still have to prove owner identity,
prompt IDs, generated IDs, and decoded text before any execution PR can be
considered.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-219-no-bias-same-callsite-receipt-emitter.json
decision = same_callsite_candidate_receipt_emitter_defined_fail_closed
reason = same_callsite_candidate_off_on_receipts_incomplete
remaining_blocker = fresh_candidate_off_on_strict_receipts
boundary = bitnet_transformer::DenseLinearNoBiasSameCallsiteCandidateReceiptEmitterBoundary
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime, does not claim generated-ID preservation for candidate-on execution,
and does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. SLM-CPU-220 can consume
this boundary to capture fresh candidate-off/candidate-on receipts or record
the exact strict-receipt artifact blocker.

## SLM-CPU-220 No-Bias Candidate-Off/On Strict Receipt Boundary

SLM-CPU-220 consumes the SLM-CPU-219 same-callsite receipt-emitter boundary and
records the strict candidate-off/candidate-on evidence gate that must pass
before any no-bias `FeedForward::apply_linear` execution PR. The owner callsite
boundary is present, but the fresh off/on strict artifact pair is not present
yet, the pair does not bind owner/callsite identity, and prompt/generated/text
preservation is not proven by same-callsite artifacts.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-220-no-bias-candidate-off-on-strict-receipts.json
decision = same_callsite_candidate_off_on_strict_receipts_blocked_fail_closed
reason = same_callsite_candidate_off_on_strict_receipt_artifacts_incomplete
remaining_blocker = candidate_on_strict_receipt_artifact
boundary = bitnet_transformer::DenseLinearNoBiasSameCallsiteCandidateOffOnStrictReceiptBoundary
same_callsite_receipt_emitter_ready = true
candidate_off_strict_receipt_artifact_present = false
candidate_on_strict_receipt_artifact_present = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not enable candidate execution, does not change the default
runtime, does not claim candidate-on generated-ID preservation, and does not
make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. The next safe slice must
either capture the required strict off/on artifact pair or keep the execution
attempt fail-closed with the exact missing evidence recorded.

## SLM-CPU-221 No-Bias Receipt-Gated Candidate Execution Blocker

SLM-CPU-221 consumes the SLM-CPU-220 candidate-off/on strict receipt boundary
and records the next receipt-gated execution preflight. The execution attempt
is still blocked because the same-callsite candidate-off/candidate-on strict
artifact pair is incomplete. The boundary can model a future explicit
candidate attempt, but normal inference and candidate execution both remain
runtime-disabled here.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-221-no-bias-receipt-gated-candidate-execution-blocker.json
decision = blocked_fail_closed
reason = off_on_strict_receipt_boundary_incomplete
remaining_blocker = same_callsite_candidate_off_on_strict_receipts
boundary = bitnet_transformer::DenseLinearNoBiasReceiptGatedCandidateExecutionBoundary
off_on_strict_receipt_boundary_ready = false
candidate_execution_attempt_allowed = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim generated-ID preservation for a candidate-on
path, and does not make allocation, timing, speedup, sustained-throughput,
Q4/Q5, server/accelerator, Qwen3.5, or BitNet QK256 claims.

## SLM-CPU-222 No-Bias Strict Receipt Artifact Pair Boundary

SLM-CPU-222 consumes the SLM-CPU-221 receipt-gated candidate execution boundary
and records the strict same-callsite artifact-pair requirements that must be met
before a later no-bias `FeedForward::apply_linear` execution attempt can be
considered. The current artifact remains fail-closed because the receipt-gated
candidate execution boundary is still incomplete; no off/on artifact pair is
present or bound to the gate, descriptor, owner/callsite, prompt, generated, and
decoded-text identities.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-222-no-bias-strict-receipt-artifact-pair.json
decision = blocked_fail_closed
reason = receipt_gated_candidate_execution_boundary_incomplete
remaining_blocker = receipt_gated_candidate_execution_boundary
boundary = bitnet_transformer::DenseLinearNoBiasStrictReceiptArtifactPairBoundary
receipt_gated_candidate_execution_boundary_ready = false
candidate_off_strict_receipt_artifact_present = false
candidate_on_strict_receipt_artifact_present = false
candidate_execution_attempt_allowed = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim candidate-on generated-ID preservation, and
does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. The next safe slice must
produce the missing strict candidate-off/candidate-on same-callsite artifact
pair or keep runtime candidate execution fail-closed with the exact missing
artifact evidence recorded.

## SLM-CPU-223 No-Bias Strict Artifact Capture Boundary

SLM-CPU-223 consumes the SLM-CPU-222 strict receipt artifact-pair boundary and
records the concrete capture blocker for the same-callsite candidate-off/on
artifact pair. The capture remains fail-closed because the artifact-pair
boundary is not ready: the receipt-gated candidate execution boundary is still
incomplete, and neither candidate-off nor candidate-on strict capture artifact
has been validated for the bounded Qwen3/Qwen2.5 Q8_0
`feed_forward.down_proj` scope.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-223-no-bias-strict-artifact-capture.json
decision = blocked_fail_closed
reason = strict_receipt_artifact_pair_boundary_incomplete
remaining_blocker = strict_receipt_artifact_pair_boundary
boundary = bitnet_transformer::DenseLinearNoBiasStrictArtifactCaptureBoundary
strict_receipt_artifact_pair_boundary_ready = false
candidate_off_capture_artifact_validated = false
candidate_on_capture_artifact_validated = false
candidate_execution_prereqs_complete = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim candidate-on generated-ID preservation, and
does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. A later tracked slice must
produce the validated strict capture artifact pair before any separately gated
candidate execution PR can be considered.

## SLM-CPU-224 No-Bias Strict Capture Artifact Pair Boundary

SLM-CPU-224 consumes the SLM-CPU-223 strict artifact-capture blocker and records
the next strict capture artifact-pair boundary for the same-callsite
candidate-off/on evidence. The pair remains fail-closed because the strict
artifact-capture boundary is not ready: the strict receipt artifact-pair
boundary is still incomplete, neither strict capture artifact exists, and the
candidate-off/on capture commands and identity bindings have not been recorded.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-224-no-bias-strict-capture-artifact-pair.json
decision = blocked_fail_closed
reason = strict_artifact_capture_boundary_incomplete
remaining_blocker = strict_artifact_capture_boundary
boundary = bitnet_transformer::DenseLinearNoBiasStrictCaptureArtifactPairBoundary
strict_artifact_capture_boundary_ready = false
candidate_off_strict_capture_artifact_present = false
candidate_on_strict_capture_artifact_present = false
candidate_off_capture_command_recorded = false
candidate_on_capture_command_recorded = false
strict_capture_artifact_pair_validated = false
candidate_execution_prereqs_complete = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim candidate-on generated-ID preservation, and
does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. A later tracked slice must
produce the validated strict capture artifacts before any separately gated
candidate execution PR can be considered.

## SLM-CPU-225 No-Bias Strict Capture Prerequisite

SLM-CPU-225 consumes the SLM-CPU-224 strict capture artifact-pair boundary and
records the exact prerequisite that still blocks candidate-on no-bias
`FeedForward::apply_linear` execution. The prerequisite remains fail-closed
because the candidate-off/on strict capture pair has not validated: the capture
artifact paths and commands are still absent, and the gate, descriptor,
owner/callsite, prompt/generated/text, model/backend, CPU backend, and
fallback=false bindings are not present together in one validated pair.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-225-no-bias-strict-capture-prereq.json
decision = blocked_fail_closed
reason = strict_capture_artifact_pair_not_validated
remaining_blocker = validated_candidate_off_on_strict_capture_artifact_pair
boundary = bitnet_transformer::DenseLinearNoBiasStrictCapturePrerequisiteBoundary
strict_receipt_artifact_pair_boundary_bound = true
strict_artifact_capture_boundary_bound = true
strict_capture_artifact_pair_boundary_bound = true
strict_capture_artifact_pair_validated = false
candidate_off_strict_capture_artifact_present = false
candidate_on_strict_capture_artifact_present = false
candidate_off_capture_command_recorded = false
candidate_on_capture_command_recorded = false
strict_capture_prerequisite_ready = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim candidate-on generated-ID preservation, and
does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. A later tracked slice must
produce or ingest the validated strict capture artifact pair named by this
prerequisite before any separately gated candidate execution PR can be
considered.

## SLM-CPU-226 No-Bias Strict Capture Pair Blocker

SLM-CPU-226 consumes the SLM-CPU-225 strict capture prerequisite and records
that the validated candidate-off/on strict capture artifact pair still does not
exist for bounded Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj` evidence. The
older SLM-CPU-209 before/after warm-session pair is useful default-path
evidence, but it is not a candidate-on strict capture artifact pair: both sides
keep candidate execution disabled and do not record candidate-on capture
commands or off/on strict capture validation.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-226-no-bias-strict-capture-pair.json
decision = blocked_fail_closed
reason = candidate_off_on_strict_capture_artifact_pair_absent
remaining_blocker = validated_candidate_off_on_strict_capture_artifact_pair
boundary = bitnet_transformer::DenseLinearNoBiasStrictCaptureArtifactPairBoundary
strict_capture_prerequisite_bound = true
strict_capture_artifact_pair_validated = false
candidate_off_strict_capture_artifact_present = false
candidate_on_strict_capture_artifact_present = false
candidate_off_capture_command_recorded = false
candidate_on_capture_command_recorded = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim candidate-on generated-ID preservation, and
does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims. A later tracked slice must
define and validate the concrete candidate-off/on strict capture commands and
artifacts before any separately gated candidate execution PR can be considered.

## SLM-CPU-227 No-Bias Strict Capture Commands

SLM-CPU-227 consumes the SLM-CPU-226 strict capture pair blocker and defines the
concrete candidate-off and candidate-on strict capture command contract for
bounded Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj` evidence. The commands are
schema and receipt contract evidence only: the candidate-on command records the
explicit gate-request shape that a later artifact must capture, but candidate
execution remains disabled until a separately gated runtime PR supplies and
validates the strict capture artifact pair.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-227-no-bias-strict-capture-commands.json
decision = capture_commands_defined_fail_closed
reason = strict_capture_commands_and_schema_defined_but_candidate_on_artifacts_not_yet_captured
remaining_blocker = validated_candidate_off_on_strict_capture_artifact_pair
boundary = bitnet_transformer::DenseLinearNoBiasStrictCaptureCommandSchemaBoundary
candidate_off_capture_command_recorded = true
candidate_on_capture_command_recorded = true
candidate_off_strict_capture_artifact_present = false
candidate_on_strict_capture_artifact_present = false
strict_capture_artifact_pair_validated = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

The command contract requires each future capture artifact to bind model SHA,
strict GGUF tokenizer authority, `runtime_api=cpu`, `selected_backend=cpu-rust`,
`fallback_used=false`, prompt/generated/text digests, explicit gate identity,
descriptor identity, and `FeedForward::apply_linear` owner/callsite identity.
The candidate-off command records the gate disabled; the candidate-on command
records the gate requested, but still requires `candidate_execution_enabled=false`
until a later separately gated runtime PR enables execution with a validated
artifact pair.

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not claim candidate-on generated-ID preservation, and
does not make allocation, timing, speedup, sustained-throughput, Q4/Q5,
server/accelerator, Qwen3.5, or BitNet QK256 claims.

## SLM-CPU-228 No-Bias Strict Capture Artifacts

SLM-CPU-228 consumes the SLM-CPU-227 command/schema slice and captures the
bounded Qwen3/Qwen2.5 Q8_0 candidate-off and candidate-on warm-session receipts
for `feed_forward.down_proj`. The validated pair proves receipt identity only:
candidate-on records the explicit no-bias runtime gate request, but candidate
execution remains disabled and the selected runtime path remains
`eager_f32_candle`.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-228-no-bias-strict-capture-pair-validation.json
decision = strict_capture_pair_validated_candidate_execution_fail_closed
qwen3_candidate_off = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-slm-cpu-228-no-bias-candidate-off.json
qwen3_candidate_on = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-slm-cpu-228-no-bias-candidate-on.json
qwen25_candidate_off = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen25-slm-cpu-228-no-bias-candidate-off.json
qwen25_candidate_on = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen25-slm-cpu-228-no-bias-candidate-on.json
strict_capture_artifact_pair_validated = true
candidate_off_runtime_gate_requested_enabled = false
candidate_on_runtime_gate_requested_enabled = true
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
prompt_ids_preserved_between_candidate_off_and_candidate_on = true
generated_ids_preserved_between_candidate_off_and_candidate_on = true
decoded_text_preserved_between_candidate_off_and_candidate_on = true
```

The live CLI determinism gate requires at least one repeated prompt group, so
the SLM-CPU-228 captures repeat the two SLM-CPU-227 prompts rather than running
one instance of each prompt. That is repo command-contract correction, not a
`ripr` issue: the artifacts still preserve the same model, tokenizer, backend,
candidate role, and claim boundary.

This slice does not execute the no-bias candidate, does not change the default
runtime path, and does not make allocation, timing, speedup,
sustained-throughput, Q4/Q5, server/accelerator, Qwen3.5, or BitNet QK256
claims. The next separately gated slice may consume this validated pair before
attempting candidate execution.

## SLM-CPU-229 No-Bias Runtime Attempt Blocker

SLM-CPU-229 consumes the SLM-CPU-228 strict capture pair and records the exact
runtime blocker before any no-bias `FeedForward::apply_linear` candidate
execution can be attempted. The strict capture pair proves candidate-off/on
identity, prompt/generated/text digest preservation, `runtime_api=cpu`,
`selected_backend=cpu-rust`, and `fallback_used=false`, but the receipt-bound
selector still does not reach the dense runtime hook registry as an executable
descriptor.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-229-no-bias-runtime-attempt-blocker.json
decision = candidate_execution_attempt_blocked_fail_closed
reason = strict_capture_pair_is_validated_but_runtime_attachment_is_incomplete
remaining_blocker = receipt_bound_selector_runtime_hook_registry_attachment
boundary = bitnet_transformer::DenseLinearNoBiasRuntimeAttemptBoundary
strict_capture_artifact_pair_validated = true
explicit_candidate_execution_gate_requested = true
runtime_hook_registry_attachment_present = false
runtime_hook_descriptor_binds_selector_identity = false
runtime_hook_descriptor_binds_strict_capture_pair = false
apply_linear_dispatch_wired_to_no_bias_candidate = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, change default runtime,
claim generated-ID preservation for a candidate execution attempt, claim
allocation or timing improvement, claim speedup or sustained throughput,
broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5,
or BitNet QK256 paths.

## SLM-CPU-230 No-Bias Runtime Hook Attachment

SLM-CPU-230 consumes the SLM-CPU-229 blocker and adds a typed
`DenseLinearNoBiasRuntimeHookAttachmentBoundary`. The boundary verifies that a
receipt-bound no-bias selector descriptor can be attached to
`DenseLinearRuntimeHookRegistry` for the exact Qwen3/Qwen2.5 Q8_0
`feed_forward.down_proj` tensor only when the strict capture pair identity,
model SHA, tokenizer authority, `runtime_api=cpu`, `selected_backend=cpu-rust`,
selected path/kernel, candidate kernel, prompt/generated/text digests,
`bias_present=false`, and `fallback_used=false` match.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-230-no-bias-runtime-hook-attachment.json
decision = runtime_hook_attachment_ready_runtime_disabled
reason = receipt_bound_selector_identity_reaches_runtime_hook_registry_but_candidate_execution_remains_separate
remaining_blocker = fresh_candidate_off_on_execution_receipts
boundary = bitnet_transformer::DenseLinearNoBiasRuntimeHookAttachmentBoundary
strict_capture_artifact_pair_validated = true
explicit_candidate_execution_gate_requested = true
runtime_hook_registry_attachment_present = true
runtime_hook_descriptor_binds_selector_identity = true
runtime_hook_descriptor_binds_strict_capture_pair = true
registry_key_matches_tensor_name = true
descriptor_ready_for_apply_linear_callsite = true
candidate_execution_attempt_allowed = false
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice only proves selector identity reaches the runtime hook registry. It
does not call the no-bias candidate, does not change default runtime selection,
does not claim allocation reduction, timing improvement, speedup, sustained
throughput, Q4/Q5 support, server/accelerator execution, Qwen3.5, or BitNet
QK256 behavior. A later separately gated PR still has to capture fresh
candidate-off/candidate-on execution receipts before candidate execution can be
considered.

## SLM-CPU-231 No-Bias Candidate Execution Receipt Gate

SLM-CPU-231 consumes the SLM-CPU-230 runtime hook attachment boundary and adds a
typed `DenseLinearNoBiasCandidateExecutionReceiptGate`. The gate records that
the receipt-bound selector identity reaches `DenseLinearRuntimeHookRegistry`,
but fresh candidate-off/candidate-on execution receipts through that attachment
are still absent. It therefore keeps candidate execution fail-closed and leaves
normal inference on `eager_f32_candle`.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-231-no-bias-candidate-execution-receipt-gate.json
decision = candidate_execution_receipt_pair_blocked_fail_closed
reason = runtime_hook_attachment_ready_but_fresh_execution_receipts_are_missing
remaining_blocker = candidate_on_execution_receipt
boundary = bitnet_transformer::DenseLinearNoBiasCandidateExecutionReceiptGate
runtime_hook_attachment_ready = true
explicit_candidate_execution_gate_requested = true
runtime_hook_registry_attachment_present = true
runtime_hook_descriptor_binds_selector_identity = true
runtime_hook_descriptor_binds_strict_capture_pair = true
candidate_off_execution_receipt_present = false
candidate_on_execution_receipt_present = false
candidate_execution_receipt_pair_ready = false
candidate_execution_enabled_by_default = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This slice does not execute the no-bias candidate, does not change default
runtime selection, does not prove generated-ID preservation for a candidate
execution attempt, and does not claim allocation reduction, timing improvement,
speedup, sustained throughput, Q4/Q5 support, server/accelerator execution,
Qwen3.5, or BitNet QK256 behavior. A later separately gated PR must capture
fresh candidate-off and candidate-on execution receipts through the attached
registry descriptor before any candidate execution claim can be made.

## SLM-CPU-232 No-Bias Execution Capture Commands

SLM-CPU-232 consumes the SLM-CPU-231 candidate execution receipt gate and
defines the concrete candidate-off/candidate-on capture command contract for
the exact Qwen3/Qwen2.5 Q8_0 `feed_forward.down_proj` scope. The commands bind
the future receipts to the same model SHA, GGUF tokenizer authority, prompt and
generated output digests, CPU backend identity, runtime hook attachment
identity, and `FeedForward::apply_linear` callsite identity. They do not capture
those execution receipts in this slice, so candidate execution remains
fail-closed and normal inference stays on `eager_f32_candle`.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-232-no-bias-execution-capture-commands.json
decision = execution_capture_commands_defined_fail_closed
reason = candidate_execution_capture_command_contract_defined_but_receipts_not_captured
remaining_blocker = candidate_off_on_execution_receipt_artifacts
command = cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- --device cpu slm-warm-session
candidate_off_gate = off
candidate_on_gate = on
role_id = layers.0.feed_forward.down_proj
callsite_identity = bitnet_transformer::FeedForward::apply_linear:layers.0.feed_forward.down_proj.weight
candidate_execution_receipts_captured = false
candidate_execution_enabled_by_default = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_path = feed_forward_down_proj_no_bias_candidate
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

The next required artifacts are the Qwen3 and Qwen2.5 candidate-off and
candidate-on execution receipts plus a validation artifact proving generated
IDs, decoded text, backend/kernel identity, strict tokenizer authority,
callsite identity, and `fallback_used=false` are preserved. This slice only
defines that capture contract. It does not execute the no-bias candidate, does
not change default runtime selection, does not prove generated-ID preservation,
and does not claim allocation reduction, timing improvement, speedup, sustained
throughput, Q4/Q5 support, server/accelerator execution, Qwen3.5, or BitNet
QK256 behavior.

## SLM-CPU-233 No-Bias Execution Receipt Blocker

SLM-CPU-233 consumes the SLM-CPU-232 capture command contract and checks whether
the exact local artifact prerequisites are present before attempting the long
candidate-off/candidate-on warm-session receipt captures. The Qwen3 Q8_0 GGUF is
present and matches its pinned SHA, but the exact Qwen2.5 Q8_0 artifact required
for the cross-model acceptance gate is not present in this workspace. The slice
therefore blocks receipt capture rather than producing a one-model-only proof or
downloading a large model implicitly.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-233-no-bias-execution-receipt-blocker.json
decision = candidate_execution_receipt_capture_blocked_fail_closed
reason = required_qwen25_q8_0_model_artifact_absent_from_workspace
remaining_blocker = fresh_qwen3_qwen25_candidate_off_on_execution_receipts
qwen3_present = true
qwen3_sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
qwen25_present = false
qwen25_expected_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
candidate_execution_receipts_captured = false
candidate_execution_enabled_by_default = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
candidate_path = feed_forward_down_proj_no_bias_candidate
candidate_kernel = dense-f32-candle-linear-no-bias-candidate
```

This blocker keeps the no-bias candidate fail-closed until fresh Qwen3 and
Qwen2.5 candidate-off/candidate-on execution receipts exist for the same
`feed_forward.down_proj` callsite. It does not execute the no-bias candidate,
does not change default runtime selection, does not prove generated-ID
preservation, and does not claim allocation reduction, timing improvement,
speedup, sustained throughput, Q4/Q5 support, server/accelerator execution,
Qwen3.5, or BitNet QK256 behavior.

## SLM-CPU-234 Qwen2.5 Artifact Prerequisite

SLM-CPU-234 consumes the SLM-CPU-233 blocker and verifies the missing artifact
prerequisite without changing the tracked model tree. The exact pinned
Qwen2.5 Q8_0 GGUF is available in ignored local `target` cache directories, and
both discovered copies match the expected SHA256. No download was attempted in
this slice and no model binary is committed.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-234-qwen25-artifact-prereq.json
decision = qwen25_artifact_prerequisite_verified_from_ignored_cache
qwen25_repo = Qwen/Qwen2.5-0.5B-Instruct-GGUF
qwen25_revision = 9217f5db79a29953eb74d5343926648285ec7e67
qwen25_filename = qwen2.5-0.5b-instruct-q8_0.gguf
qwen25_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
qwen25_bytes = 675710816
usable_receipt_capture_path = target/slm-cpu-017/cache/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf
model_binary_committed = false
download_attempted_in_this_slice = false
candidate_execution_receipts_captured = false
candidate_execution_enabled_by_default = false
normal_inference_runtime_selection_enabled = false
```

The next safe slice can use the verified ignored-cache path to attempt the
fresh Qwen3/Qwen2.5 candidate-off/candidate-on execution receipts required by
the SLM-CPU-232 command contract. This slice does not execute the no-bias
candidate, does not change default runtime selection, does not prove
generated-ID preservation, and does not claim allocation reduction, timing
improvement, speedup, sustained throughput, Q4/Q5 support,
server/accelerator execution, Qwen3.5, or BitNet QK256 behavior.

## SLM-CPU-235 No-Bias Candidate-Off/On Execution Receipts

SLM-CPU-235 consumes the SLM-CPU-234 Qwen2.5 artifact prerequisite and captures
fresh strict warm-session receipts for Qwen3/Qwen2.5 Q8_0 with
`BITNET_DENSE_LINEAR_NO_BIAS_RUNTIME=off` and `on`.

```text
validation_artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-235-no-bias-execution-capture-validation.json
qwen3_candidate_off = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-slm-cpu-235-no-bias-candidate-off-execution.json
qwen3_candidate_on = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-slm-cpu-235-no-bias-candidate-on-execution.json
qwen25_candidate_off = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen25-slm-cpu-235-no-bias-candidate-off-execution.json
qwen25_candidate_on = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen25-slm-cpu-235-no-bias-candidate-on-execution.json
validation_passed = true
candidate_execution_enabled = false
normal_inference_runtime_selection_enabled = false
selected_path = eager_f32_candle
```

The receipt pair preserves behavior across the explicit gate request. Qwen3
records identical generated outputs for candidate-off and candidate-on:

```text
prompt_0_generated_ids = [17, 151645]
prompt_0_decoded_text = 2
prompt_1_generated_ids = [17, 151645]
prompt_1_decoded_text = 2
prompt_2_generated_ids = [3925, 13, 151645]
prompt_2_decoded_text = OK.
prompt_3_generated_ids = [3925, 13, 151645]
prompt_3_decoded_text = OK.
```

Qwen2.5 also records identical generated outputs for candidate-off and
candidate-on:

```text
prompt_0_generated_ids = [19, 151645]
prompt_0_decoded_text = 4
prompt_1_generated_ids = [19, 151645]
prompt_1_decoded_text = 4
prompt_2_generated_ids = [3925, 151645]
prompt_2_decoded_text = OK
prompt_3_generated_ids = [3925, 151645]
prompt_3_decoded_text = OK
```

The candidate-on receipts set `runtime_gate_requested_enabled=true`, but
candidate execution remains disabled and the selected path remains
`eager_f32_candle`. The receipts preserve model SHA, GGUF tokenizer authority,
`runtime_api=cpu`, `selected_backend=cpu-rust`, `fallback=false`, prompt IDs,
generated IDs, decoded text, selected path/kernel, candidate path/kernel,
descriptor identity, and `FeedForward::apply_linear` callsite identity.

The SLM-CPU-232 command contract includes `--require-determinism`, which
requires at least one repeated prompt group. An initial two-unique-prompt run
failed before producing a usable receipt; the committed receipts repeat each
prompt so determinism is evaluated. That is repo command-contract drift, not a
`ripr` or external tool failure.

This slice does not promote candidate execution, change the default runtime,
claim allocation reduction, claim timing improvement or speedup, broaden Q4/Q5
support, touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet
QK256/I2_S paths.

## SLM-CPU-236 No-Bias Candidate Execution Attempt Boundary

SLM-CPU-236 consumes the validated SLM-CPU-235 candidate-off/candidate-on
receipt pair and records the next runtime boundary for a real no-bias execution
attempt:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-236-no-bias-candidate-execution-attempt.json
decision = candidate_execution_attempt_blocked_fail_closed
remaining_runtime_selection_blocker = apply_linear_no_bias_candidate_dispatch
validated_slm_cpu_235_receipt_pair = true
candidate_execution_attempt_allowed = false
normal_inference_runtime_selection_enabled = false
candidate_execution_enabled = false
selected_path = eager_f32_candle
selected_kernel = dense-f32-candle-linear
```

The SLM-CPU-235 receipt pair proves that the explicit no-bias runtime gate is
receipt-visible and preserves prompt IDs, generated IDs, decoded text,
GGUF-tokenizer authority, CPU backend identity, selected eager path, candidate
path identity, and `fallback=false` while candidate execution remains disabled.

The runtime blocker is now specific. `bitnet_transformer::FeedForward::apply_linear`
currently dispatches through:

```text
qk256 raw tensor path
strict CUDA fallback guard
dense_linear_runtime_hook_boundary audit
maybe_forward_dense_q8_sidecar_linear
record_bitnet_linear_cpu_fallback
candle_nn::Linear::forward
```

The no-bias candidate function exists as
`bitnet_transformer::dense_linear_no_bias_candidate_forward`, but there is no
receipt-gated dispatch branch in `FeedForward::apply_linear` that can call it
for model execution. That missing branch is the next blocker.

The next safe slice can add the dispatch branch only if it fails closed unless
the explicit gate, receipt-bound selector, strict SLM-CPU-235 receipt-pair
identity, tensor callsite, `bias_present=false`, CPU backend, and
`fallback=false` checks all pass. The default path must remain
`eager_f32_candle` when the explicit gate is absent.

This slice does not execute the no-bias candidate, prove generated-ID
preservation for an executed candidate path, change default runtime selection,
claim allocation reduction, claim timing improvement or speedup, broaden Q4/Q5
support, touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet
QK256/I2_S paths.

## SLM-CPU-237 No-Bias Apply-Linear Dispatch Blocker

SLM-CPU-237 consumes the SLM-CPU-236 dispatch-attempt boundary and records a
more specific implementation blocker:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-237-no-bias-dispatch-blocker.json
decision = apply_linear_dispatch_blocked_fail_closed
remaining_runtime_selection_blocker = session_prompt_bound_no_bias_selector_attachment_or_per_callsite_receipt_emitter
candidate_execution_attempted = false
candidate_execution_enabled_by_default = false
default_runtime_changed = false
```

The no-bias candidate function exists, but wiring a dispatch branch directly
inside `FeedForward::apply_linear` is not receipt-safe yet. Production hook
construction in `bitnet_models::bitnet::dense_q8_runtime_hooks_from_sidecars`
sets `DenseLinearRuntimeHookDescriptor::receipt_bound_no_bias_selector=None`
for every sidecar descriptor. That hook registry is model-load scoped, while
the SLM-CPU-235 prompt/generated/text digests are warm-session prompt-scoped.

As a result, a dispatch branch added now would either never execute because the
selector is absent, or it would need stale/global identity that is not bound to
the current prompt receipt. The next safe design must choose one of two
receipt-safe identity paths:

```text
session_scoped_hook_mutation
per_callsite_candidate_receipt_emitter
```

The session-scoped option must attach and clear prompt-bound selector identity
without crossing prompt boundaries. The per-callsite option avoids mutating
model hooks and passes the prompt-bound candidate execution descriptor directly
to the exact `FeedForward::apply_linear` callsite.

This slice does not execute the no-bias candidate, prove generated-ID
preservation for an executed candidate path, change default runtime selection,
claim allocation reduction, claim timing improvement or speedup, broaden Q4/Q5
support, touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet
QK256/I2_S paths.

## SLM-CPU-238 Per-Callsite No-Bias Receipt Emitter

SLM-CPU-238 consumes the SLM-CPU-237 selector propagation blocker and records the
safer identity path for the next no-bias experiment:

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-238-per-callsite-no-bias-receipt-emitter.json
decision = per_callsite_candidate_receipt_emitter_defined_fail_closed
remaining_runtime_selection_blocker = fresh_candidate_off_on_strict_receipts_bound_to_per_callsite_emitter
candidate_execution_attempted = false
candidate_execution_enabled_by_default = false
default_runtime_changed = false
```

The per-callsite path avoids mutating model-load hook registries with
prompt-scoped identity. Instead, it binds the Qwen3/Qwen2.5 Q8_0
`feed_forward.down_proj` selector identity to the exact
`FeedForward::apply_linear` tensor callsite after warm-session prompt digests are
known. The existing `DenseLinearNoBiasPerCallsiteCandidateReceiptEmitterBoundary`
and `DenseLinearNoBiasCandidateOffOnReceiptPairGate` model the boundary while
preserving `eager_f32_candle`.

This slice does not execute the no-bias candidate or claim timing, allocation,
speedup, sustained throughput, Q4/Q5 support, server or accelerator execution,
Qwen3.5 support, or BitNet QK256/I2_S changes. The next safe slice must capture
fresh candidate-off/candidate-on receipts bound to this per-callsite emitter
before any candidate runtime dispatch can be attempted.

## SLM-CPU-239 Per-Callsite No-Bias Candidate-Off/On Receipt Blocker

SLM-CPU-239 consumes the SLM-CPU-238 per-callsite emitter boundary and records
that the existing candidate-off/candidate-on receipts are still
request-gate-only evidence. They preserve Qwen3/Qwen2.5 strict CPU identity
across `BITNET_DENSE_LINEAR_NO_BIAS_RUNTIME=off` and `on`, but the candidate-on
receipts do not execute `dense_linear_no_bias_candidate_forward`.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-239-per-callsite-no-bias-off-on-receipts.json
decision = per_callsite_candidate_off_on_receipts_blocked_fail_closed
remaining_runtime_selection_blocker = prompt_bound_per_callsite_candidate_execution_descriptor_and_apply_linear_dispatch
candidate_function_present = bitnet_transformer::dense_linear_no_bias_candidate_forward
candidate_function_wired_to_model_execution = false
feed_forward_apply_linear_no_bias_dispatch_branch_present = false
candidate_on_receipt_executes_candidate_path = false
candidate_execution_attempted = false
candidate_execution_enabled_by_default = false
default_runtime_changed = false
```

The current `FeedForward::apply_linear` path still dispatches through the
QK256 guard, dense-Q8 hook audit, optional packed-Q8 sidecar path, and
`candle_nn::Linear::forward`. It does not receive a prompt-bound per-callsite
no-bias candidate descriptor, and it has no branch that can emit a candidate-on
receipt from the exact `feed_forward.down_proj` callsite.

The next safe slice must carry prompt-bound per-callsite identity into
`FeedForward::apply_linear` and fail closed unless the explicit gate,
SLM-CPU-235 model/tokenizer/backend/fallback/digest identity, callsite
identity, `bias_present=false`, selected/candidate kernel identity, and default
eager path preservation all match. This slice does not execute the no-bias
candidate, prove generated-ID preservation for an executed candidate, change
default runtime selection, claim allocation reduction, claim timing improvement
or speedup, broaden Q4/Q5 support, touch server, GPU, NPU, OpenVINO, UHD 620,
Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-240 Per-Callsite No-Bias Dispatch Descriptor Blocker

SLM-CPU-240 consumes the SLM-CPU-239 blocker and records the exact runtime
boundary that still prevents a valid candidate-on receipt. The per-callsite
off/on identity exists, but `FeedForward::apply_linear` still has no
prompt-bound no-bias candidate descriptor argument and no fail-closed dispatch
branch that calls `dense_linear_no_bias_candidate_forward`.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-240-per-callsite-no-bias-dispatch-descriptor.json
decision = per_callsite_dispatch_descriptor_blocked_fail_closed
remaining_runtime_selection_blocker = feed_forward_apply_linear_prompt_bound_candidate_descriptor_argument
candidate_off_on_receipt_pair_gate_ready = true
prompt_bound_candidate_descriptor_argument_present = false
descriptor_identity_reaches_apply_linear_callsite = false
generated_text_digests_available_at_apply_linear = false
feed_forward_apply_linear_no_bias_dispatch_branch_present = false
candidate_execution_attempt_allowed = false
candidate_execution_enabled_by_default = false
default_runtime_changed = false
```

The structural blocker is now precise. Model-load hook registries are built
before prompt/generated/text digests exist, while `FeedForward::apply_linear`
executes before generated IDs and decoded text exist for the current session.
Mutating model-load hooks with prompt-scoped identity would risk stale identity
crossing prompt boundaries. The next safe slice is a prompt/session-scoped
per-callsite descriptor argument or equivalent callsite emitter path that binds
identity at the exact `feed_forward.down_proj` execution point and emits the
candidate-on receipt only after the candidate branch actually executes.

This slice does not execute the no-bias candidate, prove generated-ID
preservation for an executed candidate, change default runtime selection, claim
allocation reduction, claim timing improvement or speedup, broaden Q4/Q5
support, touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet
QK256/I2_S paths.

## SLM-CPU-241 Apply-Linear Callsite Descriptor Boundary

SLM-CPU-241 consumes the SLM-CPU-240 blocker and adds the first fail-closed
prompt-bound descriptor argument at the `FeedForward::apply_linear` owner. The
argument is optional, the normal `forward` and `forward_with_workspace` paths
still pass `None`, and the validator accepts only the exact
`feed_forward.down_proj` callsite identity.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-241-apply-linear-callsite-descriptor.json
decision = per_callsite_dispatch_descriptor_blocked_fail_closed
remaining_runtime_selection_blocker = prompt_bound_session_descriptor_construction
candidate_off_on_receipt_pair_gate_ready = true
prompt_bound_candidate_descriptor_argument_present = true
prompt_bound_session_descriptor_constructed = false
descriptor_identity_reaches_apply_linear_callsite_for_production_session = false
feed_forward_apply_linear_no_bias_dispatch_branch_present = false
candidate_execution_attempt_allowed = false
candidate_execution_enabled_by_default = false
default_runtime_changed = false
```

The code boundary now rejects stale or wrong-callsite descriptor identity before
any candidate dispatch can occur. It rejects gate/up projection attachment,
tensor-name mismatch, callsite identity mismatch, enabled candidate execution,
non-CPU backend identity, fallback use, or default-runtime drift. The remaining
production blocker is not a missing function parameter anymore; it is the
session-scoped descriptor construction and digest lifetime path that must pass
current prompt identity to the exact callsite without mutating model-load hook
state across prompt boundaries.

This slice does not execute `dense_linear_no_bias_candidate_forward`, emit a
valid candidate-on execution receipt, prove generated-ID preservation for an
executed candidate, change default runtime selection, claim allocation
reduction, claim timing improvement or speedup, broaden Q4/Q5 support, touch
server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-242 Prompt/Session Callsite Descriptor Boundary

SLM-CPU-242 consumes the SLM-CPU-241 apply-linear argument boundary and adds an
opt-in propagation path through `TransformerModel`, `TransformerBlock`, and
`FeedForward`. The default `forward` and `forward_with_workspace` paths still
pass no descriptor. The opt-in path targets exactly one
`feed_forward.down_proj` layer before model forward and then lets
`FeedForward::apply_linear` perform the same fail-closed callsite validation.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-05-29/qwen3-qwen25-slm-cpu-242-prompt-session-callsite-descriptor.json
decision = prompt_session_callsite_descriptor_plumbing_present_blocked_fail_closed
remaining_runtime_selection_blocker = warm_session_prompt_session_descriptor_construction_and_generated_text_receipt_emitter
opt_in_descriptor_propagation_present = true
production_warm_session_descriptor_constructed = false
descriptor_identity_reaches_apply_linear_callsite_for_production_warm_session = false
generated_text_receipt_emitter_at_apply_linear_callsite = false
candidate_execution_attempt_allowed = false
candidate_execution_enabled_by_default = false
default_runtime_changed = false
```

The remaining blocker is now narrower than the model/layer call chain: the
warm-session command still needs a current-session descriptor construction
point and a generated/text receipt emitter that binds evidence after decode
without mutating model-load hook state across prompt boundaries.

This slice does not execute `dense_linear_no_bias_candidate_forward`, emit a
valid candidate-on execution receipt, prove generated-ID preservation for an
executed candidate, change default runtime selection, claim allocation
reduction, claim timing improvement or speedup, broaden Q4/Q5 support, touch
server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-246 No-Bias Role Expansion Policy

SLM-CPU-246 consumes the SLM-CPU-245 timing/allocation envelope and defines the
next receipt-bound role expansion order without enabling additional no-bias
execution.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-06-05/qwen3-qwen25-slm-cpu-246-no-bias-role-expansion-policy.json
decision = receipt_bound_role_expansion_policy_defined_without_new_candidate_execution
only_proven_executable_no_bias_role = feed_forward.down_proj
next_receipt_target = feed_forward.up_proj
candidate_execution_expanded_to_new_roles = false
new_role_dispatch_branch_enabled = false
default_path_when_gate_absent = eager_f32_candle
```

The expansion order is:

```text
1. feed_forward.up_proj
2. feed_forward.gate_proj
3. attention.o_proj
4. attention.q_proj / attention.k_proj / attention.v_proj only where
   per-model bias policy and tensor/callsite identity are exact
```

The role manifest keeps Qwen2.5 attention q/k/v fail-closed because those
records have `bias_present=true`. Qwen3 attention q/k/v remain policy targets
only after a per-model, per-callsite receipt ladder proves exact role identity,
prompt/session descriptor binding, candidate-off/candidate-on execution,
generated-ID and decoded-text parity, and a timing/allocation envelope for that
role.

SLM-CPU-246 does not execute any new role, broaden the no-bias runtime, change
the default `eager_f32_candle` path, claim speedup, claim allocation reduction,
claim sustained throughput, broaden Q4/Q5 support, touch server, GPU, NPU,
OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

## SLM-CPU-247 Kaby Optimized Opt-In Profile Surface

SLM-CPU-247 consumes the SLM-CPU-246 role expansion policy and defines the
operator-facing `slm-warm-session --profile kaby-qwen3-q8` surface without
changing the default runtime or enabling any new no-bias execution.

```text
artifact = ci/slm-cpu/intel-i5-8250u/2026-06-05/qwen3-qwen25-slm-cpu-247-kaby-optimized-opt-in-profile.json
decision = kaby_qwen3_q8_profile_surface_defined_without_default_runtime_change
profile_id = kaby-qwen3-q8
primary_model = Qwen3-0.6B-Q8_0.gguf
second_model_proof = qwen2.5-0.5b-instruct-q8_0.gguf
runtime_api = cpu
selected_backend = cpu-rust
fallback_required = false
recommended_threads = 4
default_path_when_gate_absent = eager_f32_candle
candidate_execution_enabled_by_profile = false
fresh_hardware_receipts_captured_in_slm_cpu_247 = false
speedup_claim = false
```

When the profile is explicitly selected and no corpus or manual prompts are
provided, it supplies four bounded warm-session prompts, applies strict
GGUF/tokenizer CPU settings, selects the `qwen` template with no-think greedy
deterministic decoding, uses four threads when the caller did not request a
thread count, enables quality/determinism checks, and records the profile
contract in the aggregate receipt. User-provided prompts or a corpus still
remain the prompt source.

This slice does not capture fresh Kaby hardware receipts, execute a new
candidate path, promote the no-bias runtime, claim speedup, claim allocation
reduction, claim sustained throughput, broaden Q4/Q5 support, touch server,
GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256/I2_S paths.

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
