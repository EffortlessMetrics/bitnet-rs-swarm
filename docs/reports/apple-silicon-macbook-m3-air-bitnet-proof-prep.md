# M3 Air BitNet Proof Prep

Work item: `M3MBA-031`

This report records proof-surface preparation only. It does not record a live
M3 Air BitNet local-answer pass.

## Prepared Surface

The M3 Air BitNet CPU/NEON local-answer receipt path is scoped to:

| Field | Required value |
|---|---|
| Artifact kind | `bitnet_apple_m3_air_local_answer_corpus` |
| Requested backend | `apple-m3-air-cpu-neon` |
| Selected backend | `apple-m3-air-cpu-neon` |
| Runtime API | `cpu-neon` |
| Fallback | `false` |
| Model repo | `microsoft/bitnet-b1.58-2B-4T-gguf` |
| Model file | `ggml-model-i2_s.gguf` |
| Model SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Tokenizer authority | external tokenizer JSON, SHA256 `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`, `ggml_pre=llama-bpe` |
| Prompt template | `bitnetcpp-answer` |

Synthetic gates now require generated answer text, generated token IDs, timing
fields, tokenizer identity, accepted artifact identity, and per-case strict
backend fields before the M3 receipt can pass `mac receipts-check`.

## Claim Boundary

The M3 proof-prep gates require these local-answer boundaries:

```text
local_answer_path = true
chat_enabled = false
serve_enabled = false
full_metal_inference_claimed = false
mpsgraph_inference_claimed = false
neural_engine_claimed = false
qk256_apple_claimed = false
broad_apple_silicon_claimed = false
broad_performance_claimed = false
```

The M4 Apple Silicon implementation remains useful as a reference, but M3
receipts are not M4 Mac mini evidence and M4 receipt checks are not weakened.

## Next Item

`M3MBA-032` must run or block the accepted Microsoft 2B I2_S artifact on the
exact M3 Air CPU/NEON receipt path and commit either strict local-answer evidence
or a named blocker. This item intentionally does not create a live model receipt.
