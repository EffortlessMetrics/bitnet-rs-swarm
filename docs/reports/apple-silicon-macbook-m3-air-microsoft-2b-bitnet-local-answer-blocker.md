# Apple M3 MacBook Air Microsoft 2B BitNet Local-Answer Blocker

Date: 2026-07-10
Work item: `M3MBA-032`

## Result

The strict M3 Air Microsoft 2B I2_S local-answer receipt is blocked before
model loading. The M3 proof-prep item merged, but its prescribed CLI command is
no longer present in the current binary.

Blocker artifact:
`ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-bitnet-local-answer-blocker.json`

The exact release command completed its build and exited `2` with:

```text
error: unrecognized subcommand 'bitnet-smoke'
```

No local-answer receipt was written, so `mac receipts-check` cannot run against
the intended path.

## Command

```bash
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac bitnet-smoke --device apple-m3-air-cpu-neon --model-path ~/Library/Caches/bitnet-rs/models/microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf --tokenizer-authority microsoft/bitnet-b1.58-2B-4T --override-kv tokenizer.ggml.pre=str:llama-bpe --json-out ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-bitnet-local-answer.json --quiet
```

The CLI lists related commands, including `bitnet-proof`, `bitnet-warm`, and
`smoke`. They are not safe substitutes: the current BitNet smoke and proof
paths are Apple M4 scoped, while this M3 lane requires the distinct
`bitnet_apple_m3_air_local_answer_corpus` receipt shape.

## Exact Attempt Context

- Host: MacBook Air `Mac15,13`, Apple M3, arm64, 16 GiB memory
- macOS: 26.5.1 build `25F80`
- Free cache-volume space after attempt: 252,345,276 KiB
- Model: official Microsoft BitNet 2B I2_S `ggml-model-i2_s.gguf`
- Model SHA-256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
- Tokenizer authority: `microsoft/bitnet-b1.58-2B-4T` revision
  `04c3b9ad9361b824064a1f25ea60a8be9599b127`, `tokenizer.json` SHA-256
  `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`
- Required pre-tokenizer override: `tokenizer.ggml.pre=str:llama-bpe`

No model binary is committed.

## Missing Evidence

Because dispatch failed before loading the model or tokenizer, this item did
not record generated text, generated token IDs, per-case timing, strict backend
identity, `fallback_used=false`, an answer-gate result, or receipt-check output.

## Unblock

Restore the M3-only strict local-answer command (or add an explicitly
compatible replacement) that emits the required
`bitnet_apple_m3_air_local_answer_corpus` receipt and passes `mac receipts-check`.
The repair must preserve M3/M4 separation, the real model and external tokenizer
requirements, strict CPU/NEON identity, no fallback, and disabled chat, serve,
Metal, MPSGraph, Neural Engine, QK256, M4-proof, and broad-claim paths.

Tracked in [issue #1689](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1689).

## Claim Boundary

This is a blocker report only. It does not claim that a BitNet local answer
passed, that the M3 proves M4 behavior, or that any accelerator, chat, serve,
QK256, quality, or performance capability is enabled.
