# M3 MacBook Air BitNet Resident Scorecard

Status: validated local-only scorecard UX for `M3MBA-038`.

## Operator path

Generate a fresh measured receipt with the existing strict resident corpus, then
render it without another model run or download:

```bash
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  answer-corpus --device apple-m3-air-cpu-neon --resident-session \
  --fail-on-quality --json-out target/m3-bitnet-resident-answer.json

target/release/bitnet mac m3-resident-scorecard \
  --resident-receipt target/m3-bitnet-resident-answer.json

target/release/bitnet mac receipts-check \
  target/m3-bitnet-resident-scorecard.json --json
```

The renderer defaults to the committed cold receipt at
`ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/microsoft-2b-bitnet-local-answer.json`.
It writes `target/m3-bitnet-resident-scorecard.json` and
`target/m3-bitnet-resident-scorecard.md`. Both source receipt paths and SHA-256
digests are included in the derived scorecard.

## Measured comparison

The checked resident receipt from the M3MBA-035 run is
`target/m3-bitnet-resident-answer.json`. It preserves all five answers and all
14 generated token IDs from the committed cold baseline, with 5/5 answer gates,
`fallback=false`, one model load, one tokenizer load, and a fresh KV cache for
each case.

| Metric | Committed cold baseline | Resident run |
| --- | ---: | ---: |
| Full process/corpus wall | 82.901 s | 61.436 s |
| Model load total | 28.737 s | 8.950 s |
| Tokenizer load total | 1.087 s | 0.201 s |
| Prefill total | 43.014 s | 42.766 s |
| Prefill mean | 8.603 s | 8.553 s |
| Mean time to first token | 9.400 s | 9.258 s |
| Mean steady decode | 1.471 tok/s | 1.475 tok/s |

The full-wall comparator is the process wall recorded by the committed
M3MBA-035 report, not the sum of per-case latencies. The resident run is 25.9%
faster on that exact comparison. Peak/RSS memory is not exposed by the current
answer-corpus receipt, so the scorecard reports memory as unavailable instead
of inferring it from machine capacity.

## Friction and next target

Setup uses already accepted local model and strict tokenizer artifacts; the
scorecard itself performs only local receipt reads, validation, hashing, and
rendering. The measured model command requires a release build. Build/setup
duration was not captured by the source receipts and is reported as unmeasured,
not estimated.

Exactly one next optimization target is named: **prompt prefill latency**.
Resident prefill occupies 42.766 s, or 81.8% of measured case-generation time,
while steady decode is effectively unchanged from cold. The comparator is the
same ordered five-case corpus with the same device, model, tokenizer, greedy
answers, generated token IDs, load accounting, and KV isolation, comparing
aggregate and per-case `timing.prefill_ms` plus `timing.first_token_ms`.

This evidence applies only to the exact M3 MacBook Air CPU/NEON profile. It is
not M4, broad Apple Silicon, Metal, MPSGraph, Neural Engine, QK256 acceleration,
chat, serve, or server-performance evidence.
