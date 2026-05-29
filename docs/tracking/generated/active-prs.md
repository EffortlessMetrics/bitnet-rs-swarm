<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-POWER-006A | #956 | `codex/lunar-lake/LNL258V-POWER-006A-battery-preflight-harness` | Add a no-inference Lunar Lake low-power preflight/run-harness command and runbook update that fails closed on AC/charging, records power scheme, battery status, estimated charge, and thermal sensor availability honestly, names the required power-mode/route/profile matrix and receipt fields, and prevents route/profile model inference unless battery preflight has passed. Preserve no route promotion, speedup claim, power-advantage claim, native accelerator claim, or BitNet QK256/I2_S behavior-change claim. |
