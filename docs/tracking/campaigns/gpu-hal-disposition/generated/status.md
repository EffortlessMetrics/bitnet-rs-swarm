<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# GPU HAL Disposition Campaign Status

- Campaign: `gpu-hal-disposition`
- State: `active`
- Objective: Land durable governance artifacts (ADR-0003, BITNET-PROP-0019,
BITNET-SPEC-GPU-HAL-REFERENCE-LAYER, design doc) that fix the disposition of
crates/bitnet-gpu-hal/ as a retained read-only reference crate and stop the
crate from being re-litigated by future agents. Docs-only; no runtime changes.


## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| GH-DISP-001 | pr_open | #1648 | `codex/gpu-hal/GH-DISP-001-contract-rails` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Land ADR-0003, BITNET-PROP-0019, BITNET-SPEC-GPU-HAL-REFERENCE-LAYER, this campaign manifest, and the implementation plan, all cross-linked by stable ID. ADR-0003 status lands as Proposed (promoted to Accepted in GH-DISP-003). |
| GH-DISP-002 | ready | TBD | `codex/gpu-hal/GH-DISP-002-design-doc` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Land docs/reference/gpu-hal-design.md as the canonical human-readable reference: origin (Copilot 2026-02-28 burst, no ADR/owner/handoff), the parity table (real path has every HAL capability), the 26 orphan files, the 75 misleading headers, the drift modules, and the forward guidance. Links to ADR-0003, BITNET-PROP-0019, and the spec. |
| GH-DISP-003 | ready | TBD | `codex/gpu-hal/GH-DISP-003-closeout` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Promote ADR-0003 to Accepted. Add ADR-0003 to docs/adr/README.md. Mark this campaign complete. Close issue #1639 with a comment pointing to ADR-0003. |

## Hard Constraints

- Do not change runtime code.
- Do not add or remove the crate from the workspace.
- Do not promote, demote, or create any support-tier claim.
- Do not change CI lane routing or policy ledgers.
- Do not wire any consumer onto bitnet-gpu-hal.
- A future change to the disposition MUST arrive as a superseding ADR.
