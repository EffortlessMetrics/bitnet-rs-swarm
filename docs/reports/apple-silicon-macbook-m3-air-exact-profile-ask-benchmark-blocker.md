# M3 Air exact-profile ask and benchmark blocker

The explicit M3 CPU/NEON label is fail-closed in both exact-profile wrappers.

```text
mac ask: routes the supported Mac local-answer path through --device apple-m4-cpu-neon; requested --device apple-m3-air-cpu-neon
mac benchmark: routes the dense SLM benchmark path through --device apple-m4-cpu-neon; requested --device apple-m3-air-cpu-neon
```

Both commands exited before model execution, generated output, or receipt
creation. The BitNet ask probe supplied the accepted Microsoft I2_S GGUF and
strict external tokenizer; the calibration benchmark required no model work.
Neither command fell back to CPU or another backend.

The narrow unblock is to permit the existing M3 CPU/NEON identity in each
wrapper while preserving their model-family gates, requested/selected backend,
`runtime_api=cpu-neon`, `fallback_used=false`, receipt validation, and all
unsupported-feature boundaries. No M4, Metal, MPSGraph, Neural Engine, QK256,
chat, serve, broad Apple Silicon, or performance claim follows from this
blocker.
