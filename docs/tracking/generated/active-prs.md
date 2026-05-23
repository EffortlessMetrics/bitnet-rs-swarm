<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-POWER-014 | #516 | `codex/lunar-lake/LNL258V-POWER-014-low-power-plan-energy-proxy-validity` | Make the no-inference Lunar Lake low_power battery-run plan surface POWER-013 energy-proxy validity explicitly: whether a valid battery-mode energy proxy is recorded, whether only an attempted AC-only proxy exists, the source receipt for either state, and blocker text that prevents operators from treating AC-only attempts as low_power evidence. Refresh only the low-power plan artifact and tracker state while keeping LNL258V-POWER-006 blocked and preserving no inference, route promotion, speedup, power-advantage, battery-mode evidence, measured-temperature, native accelerator, or BitNet QK256/I2_S behavior-change claims. |
