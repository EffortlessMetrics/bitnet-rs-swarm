# A770-162 full-projection packet contract

## Scope

A770-162 carries a complete, opt-in packed-QK256 source packet from the
diagnostic replay path through transformer/model context into the JSON receipt.
It does not change production dispatch and does not execute whole-matrix A770
replay. The packet is diagnostic-only and is intended to unblock the next
whole-projection replay slice from issue #1895.

The packet records:

- activation row index, quantized activation row, activation sum, and scale bits;
- logical output rows and input columns;
- packed row stride and `full_projection_output_rows` scope;
- inline weight scale bits;
- every packed QK256 byte for the projection.

The logical shape follows the loader contract. The GGUF metadata for K/V is
`[2560, 640]`, while the logical packed kernel shape is `[640, 2560]`.

## Local physical A770 proof

The proof ran on the local Intel Arc A770 with the official model from the
source repository. It used the current branch binary and one corpus case:

```text
case: a770_summary_seed770024_keywords_014
layer: 0
backend: intel-a770-opencl
runtime_api: opencl
fallback_used: false
scope: full_projection_output_rows
model_sha256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
```

The three role-filtered receipts were written under the ignored `target/`
directory. Their complete packet hashes are:

| projection | logical shape | row stride | packed bytes | SHA-256 of packed rows |
| --- | ---: | ---: | ---: | --- |
| q_proj | `[2560, 2560]` | 640 | 1,638,400 | `81bf7c8770a7808f2f37b857d663f8a46502826a24ad7fa820528f62da8d5dda` |
| k_proj | `[640, 2560]` | 640 | 409,600 | `11e5ec1d33d765c0d824a0d1a145b1ee9aee2f5cc47e35c2448a3fd8feca6a3d` |
| v_proj | `[640, 2560]` | 640 | 409,600 | `331d038487a1f63266b6218239d0bfc0580b079fef1b462fb09d4d3924264801` |

These hashes match the exact logical raw packed payloads read from the
official model, confirming that the packet carries physical model bytes rather
than bytes synthesized from the existing focused-row evidence.

## Claim boundary

This receipt proves source-packet materialization and selected-device identity
for the bounded diagnostic run. It does not prove CPU/A770 projection parity,
complete layer correctness, answer readiness, residency, performance, or a
production QK256 policy change.
