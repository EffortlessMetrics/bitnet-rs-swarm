<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-050 | #639 | `codex/intel-a770/A770-050-host-device-div-mul-replay` | Use the committed A770-049 QK256 device math-mode frontier to compare host replay f32 div/mul rounding against selected A770 OpenCL device div/mul behavior, classifying host replay mismatch, device default div-then-mul, device optimized div-then-mul, volatile/reassociation, host-policy match, unmatched, clean, or missing context without changing production QK256 dispatch, answer scoring, sampling, CPU/A770 parity, strict answer readiness, broad A770 quality, residency, speed, trusted-partial acceleration, or full BitNet inference claims. |
