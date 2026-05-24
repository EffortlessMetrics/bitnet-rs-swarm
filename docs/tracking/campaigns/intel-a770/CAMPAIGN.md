# Intel Arc A770 Campaign

Campaign ID: `intel-a770`

Status: active

## Objective

Make the Intel Arc A770 a receipt-backed OpenCL-first BitNet acceleration lane. OpenVINO GPU is a reference lane and must not be used as native OpenCL proof. Uncommitted transcript evidence cannot promote committed A770 claims.

## End State

- A770 backend identity is distinct from generic OpenCL, Intel NPU, Arc 140V, CUDA, and CPU fallback.
- Tracker, route matrix, kernel matrix, model contract, and claim ledger agree on the committed proof level.
- Runtime probe records OpenCL, Level Zero, OpenVINO GPU, PCI, VRAM, ReBAR, and render-node facts.
- Tiny OpenCL smoke executes with fallback=false.
- CPU/OpenCL parity exists for one kernel or subgraph.
- Receipts preserve selected device identity before performance claims.

## Hard Constraints

- OpenCL-first for native A770 proof.
- Uncommitted transcript evidence cannot promote A770 claims.
- OpenVINO GPU is reference only.
- Generic OpenCL and Arc 140V proof cannot count as A770 proof.
- CPU fallback cannot count as A770 execution.
- Performance claims require driver, PCIe, ReBAR, VRAM, power, and thermal context.

## Work Items

| Work item | Status | Notes |
| --- | --- | --- |
| A770-000 | merged | Reconcile committed tracker, matrices, claim ledger, and model contract before runtime work. |
| A770-003 | merged | Preserve selected-device identity after reconciliation. |
| A770-004 | merged | Add runtime probe. |
| A770-005 | merged | Tiny selected-device OpenCL smoke on real A770 DEV_56A0, without BitNet inference claims. |
| A770-006 | merged | Add selected-device OpenCL `matmul_i2s` CPU parity. |
| A770-007 | merged | Record receipt identity. |
| A770-006R | merged | Refresh the `matmul_i2s` parity fixture with explicit activation and packed-weight operand ordering before benchmark-baseline work. |
| A770-008 | merged | Record diagnostic benchmark-baseline timing for the selected-device `matmul_i2s` parity fixture without speedup or production BitNet claims. |
| A770-009 | merged | Add selected-device A770 OpenCL parity for grouped QK256 I2_S bytes with prequantized I8_S activation scale/sum correction, still fixture-only. |
| A770-010 | merged | Make strict A770 OpenCL QK256 dispatch requests fail closed while the production OpenCL QK256 runtime is not wired, and record non-strict A770 requests as CPU fallback/not-routed evidence. |
| A770-011 | merged | Wire a first inline-scaled selected-device A770 OpenCL QK256 dispatch candidate while keeping CPU activation quantization, quality, residency, speed, and full-inference claims closed. |
| A770-012 | merged | Commit a strict selected-device A770 OpenCL QK256 dispatch receipt and wire it into diagnostic route/kernel matrices without promoting inference, quality, residency, speed, or trusted-partial claims. |
| A770-013 | merged | Wire the CLI OpenCL feature into the real BitNet model/QK256 dispatch stack and add strict A770 route receipt fields without promoting inference, quality, residency, speed, or trusted-partial claims. |
| A770-014 | merged | Align the official BitNet model contract route matrix with the committed A770 diagnostic QK256 route without promoting answer quality, residency, speed, or trusted-partial claims. |
| A770-015 | merged | Add an A770 OpenCL answer-corpus route contract for seeded prompt evidence while keeping live execution, answer quality, residency, speed, and trusted-partial claims closed. |
| A770-016 | merged | Bind A770 OpenCL answer-corpus child runs to the committed BitNet model contract and diagnostic QK256 route metadata without promoting live answer quality, residency, speed, or full-inference claims. |
| A770-017 | merged | Align startup OpenCL runtime detection with the in-process dynamic OpenCL probe so A770 answer-corpus child runs can reach the committed diagnostic route without depending on `clinfo`. |
| A770-018 | merged | Record the first committed live five-case A770 OpenCL answer-corpus diagnostic receipt for the official BitNet 2B I2_S model, keeping reference parity, broad answer quality, residency, speed, and trusted-partial claims closed. |
| A770-019 | merged | Restore the OpenCL runtime availability build by matching the selected-device runtime probe wrapper type, keeping A770 answer quality, parity, residency, speed, and completion claims closed. |
| A770-020 | merged | Record same-box AMD 5700X AVX2 CPU and Intel A770 OpenCL answer-corpus receipts with one-step top-k logits, compare them with the answer-parity tool, and preserve the first CPU/A770 logits divergence without claiming exact parity. |
| A770-021 | merged | Add a compact logits top-k frontier to the CPU/A770 answer-parity receipt, separating same-output top-k drift from generated-output drift without changing runtime math or promoting parity. |
| A770-022 | merged | Record focused multi-step CPU/A770 logits for the generated-output divergent `yes_no_water` case and classify whether the first generated token mismatch has logit context. |
| A770-023 | merged | Add compact first-mismatch cross-chosen logit-margin evidence for the generated-output divergent `yes_no_water` case. |
| A770-024 | merged | Add a seeded A770 BitNet answer-readiness corpus contract before the next live CPU/A770 quality and parity run. |
| A770-025 | merged | Record the live seeded CPU AVX2 versus Intel A770 OpenCL answer-readiness run, including quality failures and parity divergence without promoting readiness or parity. |
| A770-026 | merged | Classify the A770-025 answer-readiness quality failures and CPU/A770 parity frontier before any scorer or runtime change; merged in #336. |
| A770-027 | merged | Record focused CPU/A770 logit context for the two generated-output divergent answer-readiness cases; merged in #344. |
| A770-028 | merged | Normalize only the punctuation/casing scoring cases identified by A770-026, rerun CPU/A770 readiness receipts, and keep remaining failures and parity divergence non-promoting. |
| A770-029 | merged | Repair the five shared prompt/scoring content failures left by A770-028, rerun CPU/A770 readiness receipts, and keep CPU/A770 parity divergence non-promoting. |
| A770-030 | merged | Record focused CPU/A770 multi-step logits for the remaining A770-029 summary generated-output divergence under corpus v1.0.2; merged in #388. |
| A770-031 | merged | Add a compact first-mismatch argmax-source frontier for the A770-030 summary divergence, preserving missing internal QK256/output-head context instead of promoting a runtime fix; merged in #398. |
| A770-032 | merged | Add compact first-mismatch internal logit source context for the A770-030/A770-031 summary divergence, classifying the live source as hidden operand drift before output-head accumulation without promoting a runtime fix; merged in #418. |
| A770-033 | merged | Add compact hidden-state source context for the A770-032 hidden-operand drift, classifying the live source as model.forward output drift before last-hidden extraction without promoting a runtime fix; merged in #439. |
| A770-034 | merged | Add compact model-forward source context for the A770-033 model.forward drift, classifying the live source as prior layer output drift before final norm without promoting a runtime fix; merged in #454. |
| A770-035 | merged | Add compact final transformer block source context for the A770-034 prior-layer drift, classifying the live source as final block input drift before final-block attention/FFN without promoting a runtime fix; merged in #463. |
| A770-036 | merged | Add compact penultimate transformer block source context for the A770-035 final-block-input drift, classifying whether the live source is already present at penultimate block input or introduced by penultimate attention/FFN boundaries without promoting a runtime fix; merged in #478. |
| A770-037 | merged | #488 added compact antepenultimate transformer block source context for the A770-036 penultimate-block-input drift; the live receipt classifies the remaining generated-output mismatch as already present at antepenultimate block input, without changing runtime math, OpenCL dispatch, QK256 kernels, scoring, sampling, route promotion, parity, readiness, quality, residency, speed, or trusted-partial-acceleration claims. |
| A770-038 | merged | #495 added compact pre-antepenultimate transformer block source context for the A770-037 antepenultimate-block-input drift; the live receipt classifies the remaining generated-output mismatch as already present at pre-antepenultimate block input, without changing runtime math, OpenCL dispatch, QK256 kernels, scoring, sampling, route promotion, parity, readiness, quality, residency, speed, or trusted-partial-acceleration claims. |
| A770-039 | merged | #500 added compact earlier transformer block source context for the A770-038 pre-antepenultimate-block-input drift; the live receipt classifies the remaining generated-output mismatch as already present at earlier block input, without changing runtime math, OpenCL dispatch, QK256 kernels, scoring, sampling, route promotion, parity, readiness, quality, residency, speed, or trusted-partial-acceleration claims. |
| A770-040 | merged | #508 added compact transformer block source stack context for the A770-039 earlier-block-input drift; the live receipt localizes the earliest divergent boundary to layer 0 attention output while layer 0 block input matches, without promoting runtime math, OpenCL dispatch, QK256, scoring, sampling, parity, readiness, quality, residency, speed, or trusted-partial-acceleration claims. |
| A770-041 | merged | #517 added compact layer-0 attention output source context, routing the focused mismatch into QKV projection source context without promoting parity, readiness, quality, residency, speed, or trusted-partial claims. |
| A770-042 | merged | #532 added compact QKV projection source context for the focused layer-0 attention output drift, keeping runtime math and promotion claims closed. |
| A770-043 | merged | #552 replayed selected QKV CPU versus A770 dispatch with identical materialized inputs and raw QK256 metadata, preserving diagnostic-only claim boundaries. |
| A770-044 | merged | #565 classified the selected QK256 numeric-policy frontier without changing production QK256 dispatch or promoting answer parity. |
| A770-045 | merged | #574 inspected selected QK256 output casting, readback byte counts, serialization, and compact output summary drift. |
| A770-046 | merged | #592 captured compact selected QK256 output readback trace evidence and routed the remaining split toward device-side expression evaluation. |
| A770-047 | merged | #611 classified selected QK256 OpenCL device-side output expression behavior without changing production kernels. |
| A770-048 | merged | #624 captured bounded selected QK256 OpenCL device-side intermediates after A770-047 left an unmatched device value. |
| A770-049 | merged | #634 inspected selected OpenCL device expression and math-mode variants without promoting any production policy change. |
| A770-050 | merged | #639 compared host replay f32 div/mul rounding against selected A770 OpenCL device div/mul behavior. |
| A770-051 | merged | #643 inspected host replay f32 expression ordering against selected A770 OpenCL device div/mul behavior. |
| A770-052 | merged | #646 inspected host replay f32 codegen and operation ordering, classifying host expression variants collapsed to policy. |
| A770-053 | merged | #649 captured strict f32 barrier evidence and classified the selected row as matching the A770 device output while still not justifying a production QK256 policy change. |
| A770-054 | ready | Capture host compiler strict-f32 barrier codegen evidence before any production QK256 policy change. |

## Current Claim Boundary

Committed A770 OpenCL proof remains diagnostic. The campaign does not currently claim full BitNet inference, trusted partial acceleration, performance speedup, support-op residency, full device residency, dense SLM support, Gemma support, or native OpenCL proof from OpenVINO GPU. The selected-device QK256 scaled fixture, A770-011 dispatch candidate, A770-012 strict dispatch receipt, A770-013 CLI route receipt scaffold, A770-014 model-contract route alignment, A770-015 through A770-020 answer-corpus and CPU/A770 parity diagnostics, A770-021 through A770-033 generated-output source-frontier diagnostics, A770-034 through A770-053 transformer/QKV/QK256 source-frontier diagnostics, and queued A770-054 host compiler strict-f32 barrier evidence item do not prove GPU-resident activation quantization, broad answer quality, selected attention residency, resident KV, reference parity, strict A770 answer readiness, CPU/A770 answer parity, production QK256 policy correctness, or full BitNet inference.

## Review Policy

A770 runtime PRs are non-stackable. Do not combine A770 with Intel NPU, Arc 140V, or CPU proof changes unless the campaign manifest explicitly allows it.
