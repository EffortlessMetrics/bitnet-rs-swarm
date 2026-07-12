# Known Limitations

Status: Draft v0.3 usable-preview boundary
Owner: BitNet-rs maintainers
Created: 2026-06-05

This page lists user-facing limitations for the v0.3 usable-preview lane. It
does not promote a release, tag, model tier, backend, server state, speedup, or
quality claim. The narrower source always wins: model coverage rows, committed
receipts, hardware matrices, route matrices, and support bundles define what is
actually proven.

## Supported Preview Scope

The usable preview is limited to exact model, artifact, tokenizer, prompt,
backend, device, and receipt rows.

The current user-facing support entry points are:

- [Support Matrix](SUPPORT_MATRIX.md)
- [BitNet Capability Matrix](BITNET_CAPABILITY_MATRIX.md)
- [CUDA Capability Matrix](CUDA_CAPABILITY_MATRIX.md)
- [Apple Capability Matrix](APPLE_CAPABILITY_MATRIX.md)
- [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md)

## What Works When Exact Rows Prove It

- Build and inspect the `bitnet` CLI.
- Use `bitnet model status` to see supported, candidate, diagnostic, and
  unsupported model/device rows.
- Fetch or verify exact supported-preview model IDs named by the support
  matrix.
- Run bounded local answer paths for exact supported-preview model/device
  rows.
- Explain receipts with `bitnet receipts explain`.
- Generate issue-focused context with `bitnet support bundle`.

These are preview claims. They are not production-readiness claims.

## Model Limitations

- Official Microsoft BitNet 2B I2_S/QK256 support is scoped to exact rows and
  proof families. It does not prove TL1, TL2, BF16/GPU-int2, dense SLM, A770,
  Metal, WGPU, Vulkan, or broad GPU behavior.
- Dense SLM support is model-family specific. Qwen2.5 proof does not prove
  Qwen3, SmolLM2, Llama, Gemma, Phi, or BitNet behavior.
- Registered or structurally valid rows are not supported answer paths until
  model coverage and receipts promote them.
- Artifact loading, tokenizer discovery, and structural validation are not the
  same as coherent answer readiness.

## Device And Backend Limitations

- Generic `gpu`, generic `cuda`, hardware visibility, or a smoke test is not an
  exact-device support claim.
- `nvidia-rtx-5070-ti-cuda` support is exact-profile only and must name the
  selected backend, route, fallback state, and receipt family.
- Apple CPU/NEON, Apple Metal, MPSGraph, Neural Engine, and MacBook lanes are
  separate proof families. Do not inherit claims across them.
- OpenVINO, A770, ROCm, WGPU, Vulkan, Metal full inference, MPSGraph, and Neural
  Engine paths remain candidate or diagnostic unless their exact matrices say
  otherwise.

## Receipt And Support Limitations

- A receipt proves what ran. It does not prove a broader model family, device
  family, speed claim, server claim, or release posture.
- `fallback_used=true` is diagnostic evidence unless the exact support row says
  fallback is expected.
- Missing tokenizer or prompt authority blocks coherent-answer claims.
- Failed quality gates are failure evidence, not supported-answer evidence.
- Support bundles are intended for issue triage and should not contain secrets,
  private model files, credentials, or unrelated logs.

## Performance Limitations

- `speedup_claim=false` is the default unless an exact benchmark-qualified
  receipt and support row promote speed for the same model, device, route, and
  profile.
- CUDA execution is not speedup.
- SIMD, Metal, OpenVINO, or accelerator visibility is not benchmark-qualified
  performance.
- Memory use and throughput vary by model, backend, runtime profile, and host
  hardware unless a receipt says otherwise.

## Server Limitations

- CLI readiness is not server readiness.
- Server readiness is exact-profile only. It must name the model, device,
  backend, route, endpoint, streaming mode, receipt, and support-matrix row.
- A non-streaming server smoke does not prove broad OpenAI-compatible serving,
  streaming support, concurrency, production deployment readiness, or another
  model family.
- BitNet server readiness and dense SLM server readiness are separate claims.

## Release And Packaging Limitations

- Swarm PRs prepare promotable evidence. They do not publish crates, sign
  artifacts, tag releases, or mutate the public source repository.
- Public source release, publish, signing, and tag actions remain source-repo
  owned.
- README, quickstart, and release notes summarize current status. They do not
  override receipts, model coverage, hardware matrices, or support matrices.
- Preview artifacts should not imply crates.io, docs.rs, binary archive, or
  installer availability unless those paths are actually tested and recorded.

## Where To File Issues

Start with:

```powershell
bitnet model status --format json
bitnet receipts explain --latest --format json
bitnet support bundle --latest --format json
```

Then follow [Support Triage Guide](../tutorials/support-triage.md). Include the
support bundle JSON when possible and keep claim boundaries intact.
