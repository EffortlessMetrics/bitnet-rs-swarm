# GPU HAL Disposition implementation plan

Status: active
Owner: gpu-hal
Created: 2026-06-21
Linked proposal: [BITNET-PROP-0019 GPU HAL Disposition](../../docs/proposals/BITNET-PROP-0019-gpu-hal-disposition.md)
Linked specs: [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../../docs/specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md)
Linked ADRs: [ADR-0003 GPU HAL disposition](../../docs/adr/0003-gpu-hal-disposition.md)
Linked plan: n/a
Linked issues: [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)
Linked PRs: n/a
Support-tier impact: no support promotion; docs/spec/ADR contract only
Policy impact: no policy exception

## Scope

Land the durable governance artifacts that fix the disposition of
`crates/bitnet-gpu-hal/`. This plan is **docs-only**: no runtime code, no
crate deletion, no support-tier change, no CI-lane change. The goal is to
stop the crate from being re-litigated by recording one canonical, linked
set of contract artifacts.

## End state

- ADR-0003 is `Accepted` and linked from `docs/adr/README.md`.
- BITNET-SPEC-GPU-HAL-REFERENCE-LAYER is `accepted`.
- `docs/reference/gpu-hal-design.md` is the canonical reference for the
  crate (origin, parity table, governance gap, forward guidance).
- The `gpu-hal-disposition` campaign reaches `complete` with all work
  items `merged`.
- Issue [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)
  is closed with a pointer to ADR-0003.
- Future agents reading AGENTS.md + ADR-0003 + the design doc can answer
  "what is gpu-hal and should I touch it?" without re-investigating.

## Invariants

- Main remains releasable after every PR (docs-only, so this is trivially
  true, but stated for the contract).
- No runtime code changes.
- No crate is added or removed from the workspace.
- No support-tier promotion or demotion.
- No policy-ledger change.
- All artifacts link by stable ID and relative path; no duplicated truth.
- Generated dashboards remain consistent (run `campaign generate --check`
  after each PR).

## Landing order

1. **ADR-0003 + Proposal + Spec** (rails PR). Land the contract first; it
   is the authority everything else points to.
2. **Design doc** (`docs/reference/gpu-hal-design.md`). The human-readable
   reference; depends on the contract being in place to link to.
3. **Closeout**. Update `docs/adr/README.md` to list ADR-0003, close the
   campaign, close [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639).

The rails PR (step 1) MAY be split into separate ADR / Proposal / Spec
commits if the reviewer prefers atomicity, but they land together as one
coherent contract and should not be split across PR boundaries (the
artifacts reference each other).

## Work items

### GH-DISP-001 — Contract rails (ADR-0003 + Proposal-0019 + Spec)

Status: ready
Spec requirements:
- BITNET-SPEC-GPU-HAL-REFERENCE-LAYER/REQ-001 (stable surface)
- BITNET-SPEC-GPU-HAL-REFERENCE-LAYER/REQ-002 (no consumers)
- BITNET-SPEC-GPU-HAL-REFERENCE-LAYER/REQ-006 (compile-only proof)

Expected files:
- `docs/adr/0003-gpu-hal-disposition.md` (new)
- `docs/proposals/BITNET-PROP-0019-gpu-hal-disposition.md` (new)
- `docs/specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md` (new)
- `docs/tracking/campaigns/gpu-hal-disposition/active.toml` (new)
- `docs/tracking/campaigns/gpu-hal-disposition/CAMPAIGN.md` (new)
- `plans/gpu-hal/implementation-plan.md` (new)
- `plans/gpu-hal/README.md` (new)

Non-goals:
- No runtime code changes.
- No change to the real GPU path.

Acceptance:
- ADR-0003 status is `Accepted` (after review) or `Proposed` (landing
  state, promoted to `Accepted` in the closeout PR).
- All three artifacts cross-link by stable ID.
- Campaign manifest validates.

Proof:
```bash
cargo run --locked -p xtask --no-default-features -- campaign check gpu-hal-disposition
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Rollback:
- Revert the docs commit. No runtime effect.

Claim boundary:
- Landing the contract does not prove the crate is correct, validated, or
  integrated. It proves only that the disposition is recorded.

### GH-DISP-002 — Design reference doc

Status: ready
Blocked by: []

Expected files:
- `docs/reference/gpu-hal-design.md` (new)

Non-goals:
- No runtime code changes.
- Does not promote any claim.

Acceptance:
- The doc records: origin (Copilot 2026-02-28 burst, no ADR/owner/handoff),
  the parity table (real path has every HAL capability), the 26 orphan
  files, the 75 misleading headers, the drift modules, and the forward
  guidance (reopen only via superseding ADR).
- The doc links to ADR-0003, BITNET-PROP-0019, and the spec.

Proof:
```bash
git diff --check
# Doc links resolve:
#   ADR-0003, BITNET-PROP-0019, BITNET-SPEC-GPU-HAL-REFERENCE-LAYER
```

Rollback:
- Revert the doc commit.

Claim boundary:
- The doc is an explanation artifact. It does not change the contract; the
  ADR and spec own the contract.

### GH-DISP-003 — Closeout

Status: ready
Blocked_by: ["GH-DISP-001", "GH-DISP-002"]

Expected files:
- `docs/adr/README.md` (add ADR-0003 entry)
- `docs/tracking/campaigns/gpu-hal-disposition/active.toml` (status -> complete)
- `docs/tracking/campaigns/gpu-hal-disposition/` closeout note (optional)

Non-goals:
- No runtime code changes.

Acceptance:
- ADR-0003 listed in `docs/adr/README.md`.
- Campaign status is `complete`; all work items `merged`/`done`.
- Issue [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)
  closed with a comment pointing to ADR-0003.

Proof:
```bash
cargo run --locked -p xtask --no-default-features -- campaign check gpu-hal-disposition
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Rollback:
- Revert the closeout commit.

Claim boundary:
- Closeout records that the documentation lane landed. It does not claim
  the crate is anything more than the spec allows.

## Validation

Run these proof commands for every PR in the lane:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check gpu-hal-disposition
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If a command cannot run because of environment limitations, record the
command, reason, substitute evidence, and whether it blocks merge.

## Claim boundaries

- Do not promote any runtime or support claim.
- Do not claim the HAL is a validated GPU path.
- Do not claim the HAL is the future dispatch layer.
- Do not claim any capability the real path lacks is supplied by the HAL.
- Do not touch runtime code, the real GPU path, CI lanes, support tiers,
  or policy ledgers.
- A future change to the disposition (integration, extraction, deletion)
  MUST arrive as a superseding ADR, not as a follow-up to this plan.

## Plan principles honored

- One coherent semantic slice per work item.
- Reversible sequencing (every PR is a docs-only revert).
- Scaffolding (contract) separated from explanation (design doc) separated
  from enforcement (closeout).
- No unrelated cleanup combined with a contract landing.
- Every item has proof and a claim boundary.
