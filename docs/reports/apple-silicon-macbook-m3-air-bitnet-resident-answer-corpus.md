# M3 Air BitNet resident answer-corpus proof

## Scope

`M3MBA-035` replaces the strict M3 Air BitNet answer-corpus process-per-case
execution shape with an explicit `--resident-session` route. It loads the
accepted Microsoft I2_S GGUF and strict external tokenizer once, recreates the
KV cache for each prompt, and keeps the existing greedy prompt template,
answer gates, backend identity, and fallback boundary unchanged.

This is a local M3 Air proof only. It is not an M4, Metal, MPSGraph, Neural
Engine, QK256-on-Apple, chat, serve, or broad Apple Silicon claim.

## Evidence

Baseline: committed
`ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-bitnet-local-answer.json`.

Resident proof command:

```text
target/release/bitnet answer-corpus \
  --device apple-m3-air-cpu-neon \
  --model ~/Library/Caches/bitnet-rs/models/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf \
  --tokenizer ~/Library/Caches/bitnet-rs/tokenizers/microsoft-bitnet-2b/04c3b9ad9361b824064a1f25ea60a8be9599b127/tokenizer.json \
  --resident-session --fail-on-quality \
  --json-out target/m3-bitnet-resident-answer.json
target/release/bitnet mac receipts-check target/m3-bitnet-resident-answer.json --json
```

The checked resident receipt recorded all five prompts, fourteen generated
tokens, `apple-m3-air-cpu-neon`, `runtime_api=cpu-neon`, and
`fallback_used=false`. Every answer and generated-token-ID sequence matched the
committed five-case baseline exactly.

The resident path hashes the supplied external tokenizer before loading and
requires it to match the corpus authority
`e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`.
Its greedy sampling configuration uses `temperature=0`, `top_k=0`,
`top_p=1`, and `repetition_penalty=1`, matching the established greedy path
instead of applying a sampling-time repetition penalty.

| Measure | Child-per-case baseline | Resident session | Change |
| --- | ---: | ---: | ---: |
| Model loads | 5 | 1 | -4 |
| Tokenizer loads | 5 | 1 | -4 |
| End-to-end corpus wall time | 82.901s | 61.436s | -21.465s (-25.9%) |
| Answer gates | 5/5 | 5/5 | unchanged |
| Generated token IDs | baseline | exact match | unchanged |

The resident receipt separately reports `case_generation_total_ms=52.284s` and
`total_wall_ms=61.436s`. Per-case KV caches are recreated for prompt isolation;
the optimization is model/tokenizer reuse only. Per-case timeout observations
remain recorded after completion, so this route does not claim preemptive
in-process cancellation.

## Claim boundary

The result measures one cached local M3 Air execution of the accepted artifact.
It does not establish a speed guarantee, portability result, or a performance
comparison with dense SLM, M4 Mac mini, GPU, Metal, MPSGraph, Neural Engine, or
QK256-on-Apple routes.
