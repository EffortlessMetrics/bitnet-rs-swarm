# Apple M3 Air 0.7B 1bitLLM structural conversion

Date: 2026-07-10

Work item: `ABAS-002`

## Result

The official `1bitLLM/bitnet_b1_58-large` source at revision
`85d047191dcb224f0e04f20d26110caaf8dc1a47` still publishes safetensors and
tokenizer/config files, not a GGUF. This M3 Air probe downloaded the official
source into the local cache, SHA-256-pinned every required source file, and ran
the supported `st2gguf` F16 structural conversion in strict mode.

The derived F16 GGUF is structurally readable by the Rust inspector, but its
LayerNorm scan reports four suspicious norms. The strict real-model loader then
fails before it generates a token, on
`model.layers.8.post_attention_layernorm.weight` with RMS `0.31303`. The probe
therefore remains blocked at conversion/loader validation; it is not a reference
output, Apple backend, tokenizer/prompt-authority, or answer-readiness result.

Receipt:

- `ci/hardware/apple-silicon-macbook/2026-07-10/m3-air/1bitllm-07b-structural-conversion.json`

## Local proof

The release-built local tools completed these checks on Apple M3 / macOS 26.5.1:

```text
st2gguf --input model.safetensors --config config.json --tokenizer tokenizer.json \
  --arch bitnet-b1.58 --strict --output bitnet-b1.58-large-f16-structural.gguf

bitnet inspect bitnet-b1.58-large-f16-structural.gguf --ln-stats --json

bitnet run --model bitnet-b1.58-large-f16-structural.gguf --tokenizer tokenizer.json \
  --prompt 'Answer with a single digit: 2+2=' --max-tokens 1 --temperature 0.0 \
  --greedy --deterministic --strict-loader --strict-tokenizer
```

The conversion completed in 2.74 seconds, emitted 290 F16 tensors including 49
LayerNorm tensors, and wrote a 1,457,710,272-byte GGUF with SHA-256
`9f4d643142039f821ee606bdddc701e1648c0170603dbe78505bfe430e8d112c`.
The final command intentionally did not use `--allow-mock`; it failed closed
before generation rather than hiding the source/format incompatibility.

## Exact official source

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 2,915,408,840 | `100062646f1f85771ebe297c5e476642d171c2e0e916b2ed8d19dfbe201b4b52` |
| `tokenizer.json` | 1,843,131 | `1552daf0b59fe263a27541ac46ed74c6b4b12a3b231dfe960e9a8a157c097dd9` |
| `tokenizer.model` | 499,723 | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |
| `config.json` | 749 | `267458934bbad593586ad2632e17597d274cbd1dd1247d96208dd4dd8afe6a7f` |

The receipt also pins `tokenizer_config.json`, `special_tokens_map.json`, and
`added_tokens.json`. No model binary was committed. Local source and derived
artifacts were removed after the receipt data was captured.

## Claim boundary

This records an official-source inventory, a local F16 structural conversion,
and a strict loader failure. It does not claim coherent output, I2_S or TL1
compatibility, CPU/NEON or Metal inference, M4 Mac mini behavior, QK256 support,
tokenizer/pre-tokenizer authority, or performance.
