<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-ASK-009 | #743 | `codex/lunar-lake/LNL258V-ASK-009-openvino-python-fallback` | Make the Lunar Lake OpenVINO GenAI operator-ask path discover the prepared checkout-local target/lunar-lake-openvino-venv/Scripts/python.exe fallback after BITNET_LUNAR_LAKE_OPENVINO_PYTHON and .venv, so profile-promoted OpenVINO auto asks do not fail back to plain Python when the target venv exists. Verify the ask_normal auto route without an env override selects the profile-promoted OpenVINO GPU route with fallback_used=false and a bounded answer gate pass. Preserve no route-promotion, speedup, power-advantage, native accelerator, broad quality, model behavior, or BitNet QK256/I2_S behavior-change claim. |
