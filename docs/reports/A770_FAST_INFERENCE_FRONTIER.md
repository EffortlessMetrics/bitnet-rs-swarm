# A770 Fast Inference Frontier

## Current State

The A770 lane is useful but still diagnostic. A770-138 extends the focused
QK256 replay packet from seventy-one runnable Q/K/V targets to seventy-two:
the layer-0 through layer-23 Q/K/V trios for the committed summary-logits
first mismatch.
The selected device is Intel Arc A770 OpenCL and the replay receipts keep
`fallback_used = false`, `runtime_api = opencl`, and `claim_allowed = false`.

That is a good correctness frontier, not an inference-performance frontier. It
does not prove CPU/A770 answer parity, production QK256 dispatch promotion,
support-op residency, full device residency, speedup, trusted partial
acceleration, or full BitNet inference.

## What Still Blocks Great Inference

1. The current proof is too narrow.

   A770-138 covers seventy-two focused Q/K/V rows for one case and one first
   mismatch: the layer-0 through layer-23 Q/K/V trios. Fast inference needs the
   same kind of selected-device confidence across the remaining Q/K/V/O
   projection replay targets, MLP linears, and logits-facing paths that can
   affect generated tokens.

2. Production dispatch is not promotable yet.

   The selected-device replay matches the focused row, but the route still needs
   a manifest of replay targets and receipt-backed classifications before any
   production QK256 policy change can be claimed.

3. Answer parity remains open.

   The summary answer path still records a generated-output divergence. A fast
   backend is not useful until the one-step logits and bounded answer-corpus
   checks can show where parity holds and where it still fails.

4. Residency is still incomplete.

   Effective acceleration requires more than a fast QK256 kernel. The lane still
   needs support-op and data-movement proof for activation quantization,
   selected attention, KV, attention scores, softmax, value mix, output head, and
   host/device transfer boundaries.

5. Speed evidence is not yet claim-grade.

   Performance claims need driver, PCIe, ReBAR, VRAM, power, thermal, fallback,
   route, kernel count, and model-contract context. Until correctness and
   residency are bounded, benchmark numbers are diagnostic only.

## Path To Effective Fast Inference

1. Expand focused QK256 replay.

   The current manifest shape names each target, consumes or captures raw
   focused operands, runs selected-device A770 OpenCL production replay for
   available targets, and ledgers missing operands as blockers. The next narrow
   step is to continue burning down one remaining Q/K/V target at a time, then
   expand the same method outward.

2. Promote from rows to projections.

   After the manifest packet is clean, run projection-level replay for Q/K/V/O
   and MLP linears under the same selected-device, fallback-free receipt rules.

3. Reconnect to logits.

   Once projection replay is stable, rerun one-step logits and classify any
   remaining token-choice drift before touching answer scoring or sampling.

4. Re-run bounded answer parity.

   Use the seeded A770 answer-readiness corpus as the first quality gate. The
   expected promotion gate is not broad quality; it is a small, explicit,
   repeatable CPU reference versus A770 OpenCL parity packet.

5. Move residency one support op at a time.

   Promote only named resident operations with receipts: activation
   quantization, selected attention or a bounded attention subpath, KV handling,
   attention score computation, softmax, value mix, and output-head/logits
   where applicable.

6. Benchmark only after fallback-free correctness.

   Measure decode and prefill only when receipts prove selected A770 OpenCL
   execution, no CPU fallback, named kernel counts, model/tokenizer identity,
   and driver/platform context. Report speed only for the proven route.

## Next Work Item

After A770-138, the next honest step is one more focused replay target, not a
speed PR: capture and replay layer-24 `q_proj`, or the next remaining
`dispatch_replay_missing` target selected by the manifest, under the same
one-case, one-mismatch, selected-device, fallback-free receipt rules. Its job is
to keep widening the correctness surface before any production QK256 policy,
residency, answer-quality, or speed promotion.
