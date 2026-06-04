# Apple M3 MacBook Air Secondary Artifact Unblock Preflight

Date: 2026-06-04
Work item: `M3MBA-024`

## Result

`M3MBA-024` keeps the secondary BitNet candidates blocked, but makes the
unblock criteria explicit before any new large download. Current source probes
show no official GGUF artifact for either secondary candidate:

- `1bitLLM/bitnet_b1_58-large` still exposes the official safetensors and
  tokenizer files at revision `85d047191dcb224f0e04f20d26110caaf8dc1a47`, but
  no `.gguf` file for the `M3MBA-006` command shape.
- `1bitLLM/bitnet_b1_58-3B` still exposes three official safetensors shards and
  tokenizer files at revision `af89e318d78a70802061246bf037199d2fb97020`, but
  no TL1/TL2 `.gguf` file for the `M3MBA-007` diagnostic command shape.

Evidence ledger:

- `ci/model-artifacts/apple-m3-secondary-unblock-preflight.toml`

This is source availability, tokenizer-authority, storage, conversion, and
cleanup planning only. No model file was downloaded.

## Current Source Probes

| Candidate | Official repository | Revision | Official model bytes | Official GGUF files | Decision |
|---|---|---:|---:|---|---|
| 0.7B control | `1bitLLM/bitnet_b1_58-large` | `85d047191dcb224f0e04f20d26110caaf8dc1a47` | 2,915,408,840 | none | blocked |
| 3B TL diagnostic | `1bitLLM/bitnet_b1_58-3B` | `af89e318d78a70802061246bf037199d2fb97020` | 13,297,592,664 | none | blocked |

The probe used the Hugging Face model APIs and HEAD-only metadata for official
model files. It did not fetch model contents.

## Tokenizer Authority

Both official repositories expose tokenizer files, but neither exposes a
runner-ready GGUF artifact for the intended BitNet.cpp command shape. Tokenizer
file visibility is therefore not sufficient to unblock either candidate.

An unblock PR needs one of:

- an official runner-supported GGUF in the official repository;
- a reviewed conversion path from the official safetensors files to a
  runner-supported GGUF; or
- an explicitly approved third-party artifact with source revision, SHA-256,
  tokenizer/pre-tokenizer authority, runner command, prompt-suite output, and
  cleanup status.

Prior third-party large-family and 3B GGUF evidence remains rejected for answer
readiness, so a third-party substitution is a policy decision rather than a
silent unblock.

## Storage Gate

Current free space for the MacBook cache volume is 18,199,472 KiB. That is
above the 8 GiB hard floor but below the 25 GiB preferred floor used by this
lane.

Do not start a new secondary download while the lane is below the preferred
floor. A 0.7B safetensors download would leave about 15,352,393 KiB free, which
is still below the preferred floor. The 3B official shards total about
12,985,930 KiB and would leave about 5,213,542 KiB free, below the hard floor.

## Current Local Cache

The cache currently retains:

- `microsoft-bitnet-2b-i2s/ggml-model-i2_s.gguf`, about 1,159,964 KiB;
- `qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf`, about
  659,884 KiB.

Those retained artifacts are not evidence for the 0.7B or 3B secondary
candidates.

## Cleanup Plan

Future approved secondary artifact work must:

- record free space before download;
- hash the artifact outside the repository;
- commit metadata and receipts only;
- retain the local binary only while it is needed for the next decision; and
- record deletion or retention before the lane drops below the preferred floor.

## Claim Boundary

This report does not claim 0.7B answer readiness, 3B TL diagnostic output, 3B
I2_S support, BitNet-rs Apple backend support, Apple Metal BitNet inference,
M4 Mac mini proof, QK256 on Apple Silicon, speedup, or broad Apple Silicon
performance.
