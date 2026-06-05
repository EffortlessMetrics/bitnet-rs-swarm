<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-CPU-MATCHED-CHECK-001 | #1593 | `codex/lunar-lake/LNL258V-CPU-MATCHED-CHECK-001-checker-guard` | Add no-new-inference checker/test coverage for #1568 so CPU runtime-comparison benchmark qualification fails closed when remaining alignment gates are missing or contradicted: direct generated-token status, retokenized-token rejection, matched-profile completeness, fallback and answer-gate status, populated benchmark blockers under benchmark_qualified=true, benchmark-field consistency, and claim-boundary leakage. Use existing committed receipts or synthetic mutated receipts only, preserve #1156's closed model-format/timing-scope guard, keep context-only comparison valid when benchmark_qualified=false, and preserve no inference, receipt refresh, route-policy mutation, OpenVINO CPU promotion, CPU optimization, speedup or power claim, low_power evidence, dense-SLM-as-BitNet proof, or BitNet QK256/I2_S behavior-change claim. |
