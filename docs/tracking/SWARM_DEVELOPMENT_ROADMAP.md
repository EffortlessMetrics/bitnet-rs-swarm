# Swarm Development Roadmap

Status: active  
Owner: swarm orchestrators  
Created: 2026-05-24  
Linked proposal: n/a  
Linked specs: `docs/tracking/LANE_OWNERSHIP.md`, `docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md`, `docs/specs/BITNET-SPEC-GENERATED-TRACKING.md`  
Linked ADRs: `docs/adr/BITNET-ADR-0006-pr-closure-creates-backlog.md`  
Linked plan: n/a  
Linked issues: n/a  
Linked PRs: n/a  
Support-tier impact: none  
Policy impact: none

Scope: `EffortlessMetrics/bitnet-rs-swarm` development only.  
Out of scope: release, publish, signing, source-repo support claims, broad public support declarations.

## Purpose

This roadmap coordinates parallel development inside `bitnet-rs-swarm`.

It is not a release roadmap. It does not define public support, publishing, signing, or source-repo release readiness. Hardware lanes retain their hardware identities and campaign ownership. The roadmap defines shared swarm-development objectives, cross-lane proof surfaces, collision rules, and review priorities so owned lanes can advance in parallel without erasing each other’s state.

The durable lane ownership contract remains `docs/tracking/LANE_OWNERSHIP.md`. Campaign manifests, campaign-local `active.toml`, and append-only campaign events remain authoritative. GitHub labels are navigation metadata only.

## Current Development Model

Swarm development is parallel by design.

Each PR must declare:

- Lane
- Campaign
- Work item
- Orchestrator
- Branch
- Base main SHA
- Allowed paths
- Shared surfaces touched
- Closeout required
- Source promotion needed
- Model/hardware/proof claims added
- Claims explicitly not promoted
- Commands run
- Validation gaps
- Rollback

A roadmap item does not override campaign authority. If the roadmap and a campaign manifest conflict, fix the campaign/spec/roadmap relationship explicitly rather than silently treating the roadmap as authority.

## Pillar 1: Preserve Hardware Lane Identity

Hardware lanes are hardware-owned development streams, not interchangeable backend tickets.

Each lane owns its device identity, campaign manifest, proof artifacts, validation gaps, and rollback plan.

Proof does not transfer across:

- device,
- backend,
- route,
- model family,
- artifact,
- tokenizer/prompt authority,
- server profile,
- benchmark profile,
- fallback mode.

A770 proof does not imply CUDA proof. Apple proof does not imply Lunar Lake proof. Dense SLM proof does not imply BitNet QK256 proof. One-profile receipts do not imply broad support.

## Pillar 2: Shared Surfaces Are Coordination Points

Lane ownership remains campaign-local. Shared surfaces coordinate state across lanes.

Shared surfaces include:

- `docs/tracking/generated/**`
- `docs/tracking/generated/global-dashboard.md`
- `docs/tracking/generated/lane-dashboard.md`
- `docs/tracking/generated/blocked-items.md`
- `docs/tracking/generated/active-prs.md`
- `AGENTS.md`
- `README.md`
- `.github/**`
- `xtask/**`
- `ci/hardware/device-kernel-routing.toml`

Generated dashboards are outputs, not authority. A PR touching generated dashboards must state which campaign-local source changed and include generator/checker evidence.

If two PRs collide only on generated dashboards, preserve both campaign-source changes and regenerate.

## Pillar 3: Claim Boundaries Are Development Safety Rails

Swarm development may add diagnostics, receipts, route visibility, support-bundle fields, and hardware-local evidence. These artifacts do not automatically promote support claims.

Every hardware/proof PR must state:

- claims added,
- claims explicitly not promoted,
- exact hardware/profile scope,
- exact route scope,
- exact model-family scope,
- fallback behavior,
- validation gaps.

A development PR may improve evidence without claiming support.

## Pillar 4: CI Spend Follows Active Proof Needs

CI exists to keep swarm development moving.

Priority order:

1. current merge candidates,
2. clean ports,
3. required proof for active work items,
4. shared-surface PRs,
5. blocker recovery.

Avoid CI spend for archaeology, stale-stack churn, dashboard-only rewrites, or speculative proof unless the selected spec/campaign requires it.

## Pillar 5: Hardware and Runtime Lanes Do Not Stack By Default

Do not combine unrelated hardware identities in one PR.

Do not combine:

- A770 + CUDA,
- Apple + Lunar Lake,
- NPU + OpenCL GPU,
- hardware + server,
- hardware + model-family onboarding,
- hardware + CI-routing,

unless the campaign manifest explicitly allows the overlap.

Cross-lane integration is allowed, but it must be named as cross-lane integration and must list all affected lanes and shared surfaces.

## Pillar 6: Swarm Main Remains a Usable Development Base

The goal is not release readiness. The goal is that `bitnet-rs-swarm/main` remains a reliable integration base for parallel agents.

Swarm main should preserve:

- compiling active development surfaces,
- working generators/checkers,
- campaign state consistency,
- readable PR queue state,
- hardware-lane isolation,
- no accidental source/release claims,
- no committed generated build debris,
- no hidden fallback claims.

## Near-Term Roadmap Themes

### Theme A: Lane Ownership Hygiene

Goal: every active hardware/runtime PR has complete lane metadata.

Work:

- audit current PRs for required lane fields,
- close or restack only after content review,
- add missing campaign/work-item/orchestrator data,
- ensure shared-surface PRs name their generator/checker evidence.

Done when:

- every active PR has complete lane fields,
- ambiguous branches are renamed or superseded,
- generated-dashboard conflicts are resolved through campaign-local sources.

### Theme B: Hardware Claim Boundary Cleanup

Goal: hardware lanes can add evidence without accidentally claiming support.

Work:

- audit hardware docs for support-like language,
- convert broad claims into exact hardware/profile/route statements,
- add "claims explicitly not promoted" sections where missing,
- ensure receipts remain exact-profile scoped.

Done when:

- hardware docs distinguish evidence from claims,
- route/device/model proof boundaries are explicit,
- no one-lane proof is described as cross-hardware support.

### Theme C: Campaign-State Reliability

Goal: campaign manifests and generated dashboards remain consistent.

Work:

- prefer campaign-local source edits over dashboard edits,
- run `campaign generate --check` on shared-surface changes,
- preserve other lanes’ rows during regeneration,
- add short-lived leases when shared surfaces are touched.

Done when:

- generated tracking is reproducible,
- dashboard conflicts are resolved by regenerating from sources,
- active campaign state matches PR queue state.

### Theme D: Swarm CI Routing and Proof Economy

Goal: CI runs where it proves active work.

Work:

- keep CI focused on merge candidates and required proof,
- avoid archaeology-only reruns,
- separate shared-surface checks from hardware-local proof,
- document validation gaps when hardware is unavailable.

Done when:

- PR bodies list commands run and validation gaps,
- reruns are tied to current proof needs,
- blocked items identify whether they are CI, hardware, policy, or semantic blockers.

### Theme E: Hardware-Lane Local Roadmaps

Goal: each hardware identity maintains its own next-step plan.

Each lane should keep or add a local roadmap/status section covering:

- current hardware identity,
- current route/backend focus,
- current proof target,
- current blockers,
- receipts/evidence produced,
- claims explicitly not promoted,
- next three work items.

Candidate lane-local files:

- `docs/tracking/campaigns/intel-a770/CAMPAIGN.md`
- `docs/tracking/campaigns/apple-m4/CAMPAIGN.md`
- `docs/tracking/campaigns/intel-258v-platform/CAMPAIGN.md`
- `docs/specs/nvidia-rtx-5070-ti-roadmap.md`
- `docs/specs/apple-m4-mac-mini-roadmap.md`
- `docs/specs/intel-lunar-lake-258v-buildout-plan.md`

The global swarm roadmap should link to lane-local plans, not duplicate them.

### Theme F: Source/Release Boundary Cleanliness

Goal: swarm work stays swarm-only unless an explicit repository-boundary task says otherwise.

Work:

- mark ordinary PRs as `swarm-only`,
- avoid release/publish/signing language,
- keep source-repo promotion fields `n/a` for ordinary swarm-only PRs,
- do not mix release-boundary edits into hardware proof PRs.

Done when:

- ordinary swarm PRs do not look like release PRs,
- repo-boundary fields are explicit,
- release/publish/signing impact is absent or `n/a` for swarm-only work.

## Short Form

The swarm repo is for parallel, owned hardware/proof development. Hardware lanes keep their identities and campaign authority. The roadmap coordinates shared surfaces, proof boundaries, CI economics, generated tracking, and collision rules. It does not define releases, support claims, publishing, signing, or source-repo promotion.
