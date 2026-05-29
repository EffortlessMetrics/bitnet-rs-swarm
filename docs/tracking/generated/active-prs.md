<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-GOAL-AUDIT-077 | #964 | `codex/lunar-lake/LNL258V-GOAL-AUDIT-077-current-main` | Refresh the no-inference Lunar Lake excellence audit and prompt-to-artifact checklist to current swarm main after LNL258V-GOAL-AUDIT-076/#962 and closeout #963, so source_revision points at ae356cf344fefce0eeb54ef67ef09a511ae28227, the audit/checklist record #962/#963 as audit and tracker-closeout state only, and the live current-main validate/regress/compare plus AC-blocked low-power-harness checks are summarized without adding route evidence. Preserve LNL258V-POWER-006 as blocked and preserve no new inference, model load, fallback behavior change, route promotion, speedup, power-advantage evidence, battery-mode route samples, measured-temperature proof, native accelerator proof, broad quality, dense-SLM-as-BitNet proof, full BitNet accelerator inference, production QK256 dispatch policy, or BitNet QK256/I2_S behavior-change claim. |
