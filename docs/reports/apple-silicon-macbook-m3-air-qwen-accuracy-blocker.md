# Apple M3 MacBook Air Dense Qwen Accuracy Blocker

Date: 2026-06-05
Work item: `M3MBA-029`

## Result

The bounded M3 Air dense SLM accuracy profile is blocked before receipt
emission.

Blocker artifact:
`ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-accuracy-blocker.json`

The prescribed command selected the exact M3 Air backend with no fallback, then
the warm-session gate rejected the selected runtime API:

```text
requested_backend=apple-m3-air-cpu-neon
selected_backend=apple-m3-air-cpu-neon
runtime_api=cpu-neon
fallback_used=false
error: slm-warm-session is CPU scoped; selected runtime_api=cpu-neon
```

No `qwen-mirror-accuracy.json` receipt was written.
Running `mac receipts-check` against that intended receipt path failed with:

```text
receipt path does not exist: ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-accuracy.json
```

## Command

```bash
cargo run --release --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- mac validate --profile-set accuracy --device apple-m3-air-cpu-neon --corpus ci/quality/apple-m4-slm-quality-corpus.yaml --cache-dir ~/Library/Caches/bitnet-rs/models --json-out ci/hardware/apple-silicon-macbook/2026-05-12/m3-air/qwen-mirror-accuracy.json --quiet
```

The release build completed. The runtime command exited `1` before prompt
execution or receipt writing.

The follow-up receipt-check command also exited `1` because there was no
accuracy receipt to validate.

## Context

- Host: MacBook Air `Mac15,13`, Apple M3, 8 CPU cores, 16 GiB memory
- macOS: 26.3.1 build `25D2128`
- Power: AC power, battery 100%
- Thermal: `pmset -g therm` reported no thermal, performance, or CPU power warning level
- Free space after attempt: 10,167,876 KiB on `/System/Volumes/Data`
- Model cache: existing local Qwen2.5 0.5B Q8_0 GGUF, no model binary committed
- Model SHA-256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`
- Corpus SHA-256: `7ae59c3871c3e5fde804a92bfde3ed031521c5ce335790ec0baffdbd26684513`

Existing M3 smoke and operator receipts for the same backend record
`runtime_api=cpu`, `fallback_used=false`. The current accuracy path's startup
backend selection resolves the exact same backend label to `runtime_api=cpu-neon`
and reaches a fail-closed guard in `slm-warm-session`.

## Missing Evidence

Because the command failed before receipt emission, M3MBA-029 did not capture:

- prompt IDs for the accuracy profile;
- generated token IDs;
- decoded text;
- an `accuracy_comparison_profile` block;
- `mac receipts-check` output for `qwen-mirror-accuracy.json`.

## Unblock

The next slice should teach the M3 Air `mac validate` / `slm-warm-session`
accuracy path to treat selected `apple-m3-air-cpu-neon` CPU/NEON execution as
CPU-scoped for dense SLM warm-session receipts, while preserving exact backend
identity and `fallback_used=false`.

That unblock must not weaken M4 labels, M3/M4 separation, unsupported Metal,
MPSGraph, Neural Engine, QK256 rejection, or dense SLM versus BitNet proof
boundaries.

## Claim Boundary

This report records a named blocker only. It does not claim M3 Air dense SLM
accuracy passed, broad answer quality, BitNet behavior, M4 Mac mini replacement
evidence, Metal inference, MPSGraph inference, Neural Engine execution, QK256
support, or broad Apple Silicon performance.
