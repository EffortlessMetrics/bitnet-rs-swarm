# M3 Air CPU/NEON bottleneck review

## Evidence scope

This review uses the completed M3 Air receipts for two distinct model families:
the strict five-case Microsoft BitNet 2B local-answer corpus and the existing
Qwen dense-SLM warm-session receipts. They share an M3 CPU/NEON backend but are
not quality or throughput comparators.

## Measured BitNet path

The strict BitNet receipt passed all five answer gates with
`apple-m3-air-cpu-neon`, `runtime_api=cpu-neon`, and `fallback_used=false`.
Every case cold-loads the model: load took 5,410.258–5,963.838 ms, prefill took
5,590.268–11,366.097 ms, first-token latency was 6,384–12,199 ms, and steady
decode was 1.420–1.503 tok/s. The fixed corpus is therefore dominated by
repeated model loading and prompt prefill, not by the short decode tails.

## Context only: dense-SLM resident sessions

The Qwen M3 warm receipts are a separate dense-SLM workload, not BitNet
evidence. They demonstrate the relevant execution shape only: resident sessions
record zero per-prompt model load after setup and 2.1–2.6 s first-token samples.
They do not establish a BitNet speedup, output equivalence, or shared kernel
behavior.

## Smallest safe optimization PR

The next implementation should add a **BitNet-only resident local-answer
session** that loads the accepted GGUF and strict external tokenizer once, then
runs the unchanged fixed corpus sequentially. It must preserve the current
greedy prompt template, exact generated text/token IDs, requested and selected
M3 CPU/NEON identity, `runtime_api=cpu-neon`, and `fallback_used=false`.

Acceptance evidence for that follow-up must include a before/after M3 receipt
with the same five prompt IDs, answer-gate results, generated token IDs, model
and tokenizer hashes, per-case timing, resident-session load accounting, and no
chat, serve, Metal, MPSGraph, Neural Engine, QK256-on-Apple, M4, broad Apple
Silicon, or unmeasured speedup claim.

## Decision

Pursue resident-session reuse before changing kernels or quantization routes.
It directly targets the measured repeated-load bottleneck and has a narrow
behavior-preservation oracle: the existing strict answer corpus.
