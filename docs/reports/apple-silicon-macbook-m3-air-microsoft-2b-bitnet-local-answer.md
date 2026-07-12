# M3 Air Microsoft 2B BitNet local-answer receipt

## Result

The strict local CPU/NEON answer corpus passed on this M3 Air: 5 of 5 answer
gates passed, with 14 generated tokens and no timeout, not-run, or fallback
case. `target/release/bitnet mac receipts-check` accepted the committed receipt.

| Field | Recorded value |
| --- | --- |
| Artifact | `bitnet_apple_m3_air_local_answer_corpus` |
| Model | `microsoft/bitnet-b1.58-2B-4T-gguf` `ggml-model-i2_s.gguf` |
| Model SHA-256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Tokenizer authority | external JSON, `llama-bpe`, SHA-256 `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` |
| Backend | requested and selected `apple-m3-air-cpu-neon`; `runtime_api=cpu-neon` |
| Fallback | `false` |

## Answer-gate scorecard

| Prompt id | Output | Generated token IDs | Gate |
| --- | --- | --- | --- |
| `math_2_plus_2` | `4` | `19, 128009` | passed |
| `capital_france` | `Paris` | `60704, 128009` | passed |
| `repeat_colors` | `red blue green` | `1171, 6437, 6307, 128009` | passed |
| `say_ok` | `OK` | `4012, 128009` | passed |
| `yes_no_water` | `No. N/A` | `2822, 13, 452, 10576` | passed |

## Timing

This is a cold-per-case local receipt, not a throughput or speedup claim. Model
load ranged from 5,410.258 to 5,963.838 ms; first-token latency ranged from
6,384 to 12,199 ms; steady decode ranged from 1.420 to 1.503 tok/s.

## Retention and boundaries

The GGUF and tokenizer stay in the local cache and are not committed. The
source-controlled receipt is 64 KiB; temporary per-case child captures remain
local only. This proof does not claim chat, serve, Metal, MPSGraph, Neural
Engine, QK256-on-Apple, broad Apple Silicon behavior, or a performance gain.
